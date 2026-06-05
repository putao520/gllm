fn anonymize_layer_index(name: &str, layer_idx: usize) -> String {
    let parts: Vec<&str> = name.split('.').collect();
    let mut new_parts = Vec::new();
    let idx_str = layer_idx.to_string();

    // Use the same logic as match_tensor_role to locate the layer index
    let mut replaced = false;
    for (i, part) in parts.iter().enumerate() {
        if !replaced && *part == idx_str {
            // Check context like match_tensor_role
            if i > 0 {
                let prefix = parts[i - 1];
                if matches!(
                    prefix,
                    "layers" | "blk" | "blocks" | "h" | "layer" | "block"
                ) {
                    new_parts.push("{}");
                    replaced = true;
                    continue;
                }
            }
            // Fallback: if we didn't match prefix but exact string matches and we haven't replaced yet,
            // we might consider it. But match_tensor_role is strict about prefix.
            // If match_tensor_role found it, it must have matched the prefix check.
        }
        new_parts.push(part);
    }

    // If exact logic failed (maybe due to case sensitivity in match_tensor_role vs here?),
    // try simpler heuristic if not replaced?
    // match_tensor_role uses to_ascii_lowercase() but we have original name here.
    // Let's assume the original name preserves case but structure is standard.

    if !replaced {
        // Second pass: just find the number segment if it wasn't replaced
        new_parts.clear();
        for part in parts.iter() {
            if !replaced && *part == idx_str {
                new_parts.push("{}");
                replaced = true;
            } else {
                new_parts.push(part);
            }
        }
    }

    new_parts.join(".")
}

fn apply_tensor_derived(
    mut base: ModelConfig,
    derived: TensorDerivedConfig,
) -> ModelConfigResult<ModelConfig> {
    base.hidden_size = derived.hidden_size;
    base.num_attention_heads = derived.num_attention_heads;
    base.num_key_value_heads = derived.num_key_value_heads;
    base.num_hidden_layers = derived.num_hidden_layers;
    if let Some(intermediate_size) = derived.intermediate_size {
        base.intermediate_size = Some(intermediate_size);
    }
    base.vocab_size = derived.vocab_size;
    base.head_dim = derived.head_dim;
    base.dtype = derived.dtype;
    base.kv_cache_block_size = base.head_dim.max(base.num_key_value_heads);
    // Ω1: Update tensor map with derived patterns
    base.tensor_map = derived.tensor_map;

    if base.head_dim == 0 {
        return Err(ModelConfigError::InvalidConfig(
            "invalid tensor-derived head_dim".to_string(),
        ));
    }
    if base.kv_cache_block_size == 0 {
        return Err(ModelConfigError::InvalidConfig(
            "invalid tensor-derived kv_cache_block_size".to_string(),
        ));
    }

    Ok(base)
}

fn onnx_config_from_metadata(
    loader: &crate::loader::OnnxLoader,
) -> ModelConfigResult<Option<Value>> {
    let model_props = &loader.model().metadata.metadata_props;
    let graph_props = &loader.graph().metadata_props;

    for props in [graph_props, model_props] {
        for key in ["gllm.config", "_gllm_config"] {
            if let Some(raw) = props.get(key) {
                let parsed: Value = serde_json::from_str(raw).map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "invalid ONNX metadata json for {key}: {err}"
                    ))
                })?;
                return Ok(Some(parsed));
            }
        }
    }

    let mut root = serde_json::Map::new();
    let mut has_value = false;
    for props in [model_props, graph_props] {
        for (key, raw) in props {
            if key == "gllm.config" || key == "_gllm_config" {
                continue;
            }
            if key.trim().is_empty() {
                continue;
            }
            has_value = true;
            let parsed = parse_metadata_string_value(raw);
            insert_json_path(&mut root, key, parsed);
        }
    }

    if has_value {
        Ok(Some(Value::Object(root)))
    } else {
        Ok(None)
    }
}

fn parse_metadata_string_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
        return parsed;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    if let Ok(v) = trimmed.parse::<u64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = trimmed.parse::<i64>() {
        return Value::Number(v.into());
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        if let Some(number) = serde_json::Number::from_f64(v) {
            return Value::Number(number);
        }
    }
    Value::String(trimmed.to_string())
}

fn insert_json_path(root: &mut serde_json::Map<String, Value>, path: &str, value: Value) {
    let segments = path
        .split('.')
        .filter(|segment| !segment.trim().is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return;
    }
    insert_json_path_segments(root, &segments, value);
}

fn insert_json_path_segments(
    root: &mut serde_json::Map<String, Value>,
    segments: &[&str],
    value: Value,
) {
    if segments.len() == 1 {
        root.insert(segments[0].to_string(), value);
        return;
    }

    let entry = root
        .entry(segments[0].to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(serde_json::Map::new());
    }
    let child = entry
        .as_object_mut()
        .expect("insert_json_path ensures object");
    insert_json_path_segments(child, &segments[1..], value);
}

#[allow(dead_code)]
fn derive_dtype_size(metas: &[TensorMeta]) -> ModelConfigResult<usize> {
    let mut float_sizes = Vec::new();
    let mut all_sizes = Vec::new();

    for meta in metas {
        let Some(size) = dtype_size_from_dtype(meta.dtype) else {
            continue;
        };
        all_sizes.push(size);
        if is_floating_dtype(meta.dtype) {
            float_sizes.push(size);
        }
    }

    if !float_sizes.is_empty() {
        return unique_mode(&float_sizes, "dtype_size");
    }
    if !all_sizes.is_empty() {
        return unique_mode(&all_sizes, "dtype_size");
    }

    Err(ModelConfigError::InvalidConfig(
        "cannot derive dtype_size from tensor dtypes".to_string(),
    ))
}

/// Derive the dominant floating-point dtype string from tensor metadata.
/// Returns "f32" as default when no floating tensors are found.
fn derive_dtype(metas: &[TensorMeta]) -> ModelConfigResult<DType> {
    let mut bf16_count = 0usize;
    let mut f16_count = 0usize;
    let mut f32_count = 0usize;

    for meta in metas {
        if !is_floating_dtype(meta.dtype) {
            continue;
        }
        match meta.dtype {
            safetensors::Dtype::BF16 => bf16_count += 1,
            safetensors::Dtype::F16 => f16_count += 1,
            safetensors::Dtype::F32 => f32_count += 1,
            safetensors::Dtype::F64 => f32_count += 1, // f64 降级到 f32
            _ => {}
        }
    }

    // Pick the dominant dtype by count
    let max = bf16_count.max(f16_count).max(f32_count);
    if max == 0 {
        return Ok(DType::F32); // 默认 F32
    }
    if bf16_count == max {
        Ok(DType::BF16)
    } else if f16_count == max {
        Ok(DType::F16)
    } else {
        Ok(DType::F32)
    }
}

#[allow(dead_code)]
fn dtype_size_from_dtype(dtype: safetensors::Dtype) -> Option<usize> {
    match dtype {
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => Some(8),
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => Some(4),
        safetensors::Dtype::F16
        | safetensors::Dtype::BF16
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => Some(2),
        safetensors::Dtype::F8_E5M2
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::I8
        | safetensors::Dtype::U8
        | safetensors::Dtype::BOOL => Some(1),
        _ => None,
    }
}

fn is_floating_dtype(dtype: safetensors::Dtype) -> bool {
    matches!(
        dtype,
        safetensors::Dtype::F64
            | safetensors::Dtype::F32
            | safetensors::Dtype::F16
            | safetensors::Dtype::BF16
            | safetensors::Dtype::F8_E5M2
            | safetensors::Dtype::F8_E4M3
    )
}

#[allow(dead_code)]
fn unique_mode(values: &[usize], field: &str) -> ModelConfigResult<usize> {
    if values.is_empty() {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} cannot be derived: no candidates"
        )));
    }

    let mut counts = HashMap::<usize, usize>::new();
    for value in values {
        *counts.entry(*value).or_default() += 1;
    }

    let max_count =
        counts.values().copied().max().ok_or_else(|| {
            ModelConfigError::InvalidConfig(format!("{field} has no valid count"))
        })?;
    let mut winners = counts
        .into_iter()
        .filter_map(|(value, count)| (count == max_count).then_some(value))
        .collect::<Vec<_>>();
    winners.sort_unstable();
    if winners.len() != 1 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} is ambiguous: candidates={winners:?}"
        )));
    }
    let winner = winners[0];
    if winner == 0 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{field} resolved to invalid zero"
        )));
    }
    Ok(winner)
}

fn projection_out_dim(
    meta: &TensorMeta,
    hidden_size: usize,
    role: &str,
) -> ModelConfigResult<usize> {
    if meta.shape.len() < 2 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} must be at least 2D",
            meta.name
        )));
    }

    let a = meta.shape[0];
    let b = meta.shape[1];
    let out = if a == hidden_size && b != hidden_size {
        b
    } else if b == hidden_size && a != hidden_size {
        a
    } else if a == hidden_size && b == hidden_size {
        hidden_size
    } else {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} shape {:?} does not contain hidden_size {}",
            meta.name, meta.shape, hidden_size
        )));
    };

    if out == 0 {
        return Err(ModelConfigError::InvalidConfig(format!(
            "{role} tensor {} resolved zero output dimension",
            meta.name
        )));
    }
    Ok(out)
}

fn require_usize(value: &Value, keys: &[&str]) -> ModelConfigResult<usize> {
    find_usize(value, keys).ok_or_else(|| ModelConfigError::InvalidConfig(keys[0].to_string()))
}

fn require_gguf_usize(value: Option<u64>, field: &str) -> ModelConfigResult<usize> {
    let value = value.ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })?;
    usize::try_from(value).map_err(|_| {
        ModelConfigError::InvalidConfig(format!("GGUF metadata field overflow: {field}"))
    })
}

fn optional_gguf_usize(value: Option<u64>, field: &str) -> ModelConfigResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = usize::try_from(value).map_err(|_| {
        ModelConfigError::InvalidConfig(format!("GGUF metadata field overflow: {field}"))
    })?;
    Ok(Some(parsed))
}

fn require_gguf_f32(value: Option<f32>, field: &str) -> ModelConfigResult<f32> {
    value.filter(|v| v.is_finite()).ok_or_else(|| {
        ModelConfigError::InvalidConfig(format!("missing GGUF metadata field: {field}"))
    })
}

fn find_value<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value_at_path(value, key))
}

fn value_at_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn find_usize(value: &Value, keys: &[&str]) -> Option<usize> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn find_u32(value: &Value, keys: &[&str]) -> Option<u32> {
    find_value(value, keys)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

fn find_f32(value: &Value, keys: &[&str]) -> Option<f32> {
    find_value(value, keys).and_then(|v| v.as_f64().map(|num| num as f32))
}

fn find_f32_array(value: &Value, keys: &[&str]) -> Option<Vec<f32>> {
    let values = find_value(value, keys)?.as_array()?;
    let mut out = Vec::with_capacity(values.len());
    for item in values {
        let value = item.as_f64()? as f32;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

fn find_string(value: &Value, keys: &[&str]) -> Option<String> {
    find_value(value, keys)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn find_bool(value: &Value, keys: &[&str]) -> Option<bool> {
    find_value(value, keys).and_then(|v| {
        v.as_bool().or_else(|| {
            v.as_u64().and_then(|num| match num {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
        })
    })
}

fn rope_scaling_from_metadata_json(value: &Value) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig::default();

    if let Some(scaling) = value.get("rope_scaling") {
        if let Some(obj) = scaling.as_object() {
            if let Some(raw) = obj
                .get("type")
                .or_else(|| obj.get("scaling_type"))
                .and_then(Value::as_str)
            {
                config.scaling_type = Some(RopeScalingType::parse(raw));
            }
            config.rope_type = obj
                .get("rope_type")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            config.factor = obj.get("factor").and_then(|v| v.as_f64().map(|n| n as f32));
            let parse_array = |key: &str| -> ModelConfigResult<Option<Vec<f32>>> {
                obj.get(key)
                    .and_then(Value::as_array)
                    .map(|values| to_f32_array(values.as_slice()))
                    .transpose()
            };
            config.factors = parse_array("factors")?
                .or_else(|| parse_array("short_factor").ok().flatten())
                .or_else(|| parse_array("long_factor").ok().flatten());
            config.base = obj
                .get("base")
                .and_then(|v| v.as_f64().map(|n| n as f32))
                .filter(|v| v.is_finite() && *v > 0.0);
            config.original_max_position_embeddings = obj
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok());
            config.ext_factor = obj
                .get("ext_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.attn_factor = obj
                .get("attn_factor")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_fast = obj
                .get("beta_fast")
                .and_then(|v| v.as_f64().map(|n| n as f32));
            config.beta_slow = obj
                .get("beta_slow")
                .and_then(|v| v.as_f64().map(|n| n as f32));
        } else if let Some(factor) = scaling.as_f64().map(|n| n as f32) {
            config.factor = Some(factor);
        } else if let Some(raw) = scaling.as_str() {
            config.scaling_type = Some(RopeScalingType::parse(raw));
        }
    }

    if config.scaling_type.is_none() {
        config.scaling_type = find_string(value, &["rope_type", "rope_scaling_type"])
            .map(|v| RopeScalingType::parse(&v));
    }
    if config.factor.is_none() {
        config.factor = find_f32(value, &["rope_scaling.factor"]);
    }
    if config.factors.is_none() {
        config.factors = find_f32_array(value, &["rope_scaling.factors"]);
    }
    if config.base.is_none() {
        config.base = find_f32(value, &["rope_scaling.base", "rope_base", "rope_theta"]);
    }
    if config.original_max_position_embeddings.is_none() {
        config.original_max_position_embeddings =
            find_usize(value, &["rope_scaling.original_max_position_embeddings"]);
    }

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factor must be positive".to_string(),
            ));
        }
    }
    if let Some(factors) = &config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must contain positive finite values".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn to_f32_array(values: &[Value]) -> ModelConfigResult<Vec<f32>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let Some(v) = value.as_f64().map(|n| n as f32) else {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be numeric".to_string(),
            ));
        };
        if !v.is_finite() {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scaling.factors must be finite".to_string(),
            ));
        }
        out.push(v);
    }
    Ok(out)
}

fn rope_scaling_from_gguf(
    reader: &GgufLoader,
    arch: &str,
) -> ModelConfigResult<Option<RopeScalingConfig>> {
    let mut config = RopeScalingConfig {
        scaling_type: gguf_arch_str(reader, arch, "rope.scaling.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling"))
            .map(RopeScalingType::parse),
        rope_type: gguf_arch_str(reader, arch, "rope.type")
            .or_else(|| gguf_arch_str(reader, arch, "rope.scaling.rope_type"))
            .map(|v| v.to_string()),
        factor: gguf_arch_f32(reader, arch, "rope.scaling.factor"),
        factors: gguf_arch_array_f32(reader, arch, "rope.scaling.factors")
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.short_factor"))
            .or_else(|| gguf_arch_array_f32(reader, arch, "rope.scaling.long_factor")),
        base: gguf_arch_f32(reader, arch, "rope.scaling.base")
            .or_else(|| gguf_arch_f32(reader, arch, "rope.freq_base")),
        original_max_position_embeddings: gguf_arch_usize(
            reader,
            arch,
            "rope.scaling.original_max_position_embeddings",
        )
        .or_else(|| gguf_arch_usize(reader, arch, "rope.scaling.original_context_length")),
        ext_factor: gguf_arch_f32(reader, arch, "rope.ext_factor"),
        attn_factor: gguf_arch_f32(reader, arch, "rope.attn_factor"),
        beta_fast: gguf_arch_f32(reader, arch, "rope.beta_fast"),
        beta_slow: gguf_arch_f32(reader, arch, "rope.beta_slow"),
    };

    if let Some(factor) = config.factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factor".to_string(),
            ));
        }
    }

    if let Some(factors) = &mut config.factors {
        if factors.is_empty() || factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(ModelConfigError::InvalidConfig(
                "GGUF metadata field invalid: rope.scaling.factors".to_string(),
            ));
        }
    }

    if config.has_any_value() {
        Ok(Some(config))
    } else {
        Ok(None)
    }
}

fn gguf_arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

fn gguf_arch_u64(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<u64> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_u64(&key)
}

fn gguf_arch_usize(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<usize> {
    gguf_arch_u64(reader, arch, suffix).and_then(|v| usize::try_from(v).ok())
}

fn gguf_arch_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<f32> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_f32(&key)
}

fn gguf_arch_str<'a>(reader: &'a GgufLoader, arch: &str, suffix: &str) -> Option<&'a str> {
    let key = gguf_arch_key(arch, suffix);
    reader.get_metadata_str(&key)
}

fn gguf_arch_bool(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<bool> {
    let key = gguf_arch_key(arch, suffix);
    let value = reader.get(&key)?;
    value.as_bool().or_else(|| value.as_u64().map(|v| v != 0))
}

fn gguf_arch_array_f32(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<Vec<f32>> {
    let key = gguf_arch_key(arch, suffix);
    let array = reader.get_metadata_array(&key)?;
    let mut out = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let value = item.as_f32()?;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

/// Read a `{arch}.{suffix}` ARRAY metadata as `Vec<u8>`.
///
/// Used for Gemma 4 `attention.pattern` (0=sliding, 1=global) and similar
/// per-layer tag arrays. Items are accepted from any integer GGUF type as long
/// as they fit in u8 (0..=255). Any out-of-range or non-integer item causes the
/// whole read to return `None` (Ω1: no silent truncation).
fn gguf_arch_array_u8(reader: &GgufLoader, arch: &str, suffix: &str) -> Option<Vec<u8>> {
    let key = gguf_arch_key(arch, suffix);
    let array = reader.get_metadata_array(&key)?;
    let mut out = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let value = item.as_u64()?;
        let byte = u8::try_from(value).ok()?;
        out.push(byte);
    }
    Some(out)
}

/// Gemma 4 fallback: derive per-layer attention pattern when GGUF metadata is
/// absent. SPEC §Gemma4: every 6th layer (1-indexed) is global, others are
/// sliding-window. I.e. `(i + 1) % 6 == 0 → 1 (global)`, else `0 (sliding)`.
///
/// This is the ONLY place the default pattern is synthesised. Loader paths,
/// executor fixtures and tests must all go through this helper so the fallback
/// stays centralised (per CLAUDE.md: fallback 逻辑集中在一个函数).
pub fn derive_default_attention_pattern(num_layers: usize) -> Vec<u8> {
    (0..num_layers)
        .map(|i| if (i + 1) % 6 == 0 { 1u8 } else { 0u8 })
        .collect()
}

