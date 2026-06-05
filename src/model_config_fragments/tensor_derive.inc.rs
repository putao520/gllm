#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorDerivedConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub tensor_map: HashMap<TensorRole, String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TensorDeriveHints {
    pub head_dim: Option<usize>,
}

const VALID_HEAD_DIMS: [usize; 6] = [32, 64, 80, 96, 128, 256];

pub(crate) fn derive_config_from_tensors_with_hints<P: TensorProvider>(
    provider: &P,
    hints: TensorDeriveHints,
) -> ModelConfigResult<TensorDerivedConfig> {
    let metas: Vec<TensorMeta> = provider.iter_tensors().collect();
    if metas.is_empty() {
        return Err(ModelConfigError::InvalidConfig(
            "tensor provider returned no tensors".to_string(),
        ));
    }

    // 1. Group tensors by role
    let mut role_map: HashMap<(TensorRole, Option<usize>), &TensorMeta> = HashMap::new();
    let mut max_layer_idx = 0;
    let mut has_layers = false;

    for meta in &metas {
        if let Some((role, layer)) = match_tensor_role(&meta.name) {
            role_map.insert((role, layer), meta);
            if let Some(idx) = layer {
                if idx > max_layer_idx {
                    max_layer_idx = idx;
                }
                has_layers = true;
            }
        }
    }

    // 2. Derive Vocab Size & Hidden Size from Embedding
    // Prefer explicitly identified embedding role
    let embedding_meta = role_map
        .get(&(TensorRole::Embedding, None))
        .or_else(|| role_map.get(&(TensorRole::Embedding, Some(0))));

    let (vocab_size, hidden_size) = if let Some(meta) = embedding_meta {
        if meta.shape.len() < 2 {
            return Err(ModelConfigError::InvalidConfig(format!(
                "embedding tensor {} must be at least 2D",
                meta.name
            )));
        }
        // Heuristic: Vocab size is usually larger than hidden size
        // If they are equal, it doesn't matter for validation purposes
        let a = meta.shape[0];
        let b = meta.shape[1];
        (a.max(b), a.min(b))
    } else {
        return Err(ModelConfigError::InvalidConfig(
            "cannot derive config: no embedding tensor found".to_string(),
        ));
    };

    // 3. Derive Layer Count
    let num_hidden_layers = if has_layers { max_layer_idx + 1 } else { 0 };

    // 4. Derive Heads and Head Dim
    // Collect (q_out, k_out) for every layer to handle alternating attention
    // configurations (e.g., Gemma 4 sliding/global layers with different head counts).
    let mut layer_qk_dims: Vec<(usize, usize)> = Vec::new();
    for layer_idx in 0..num_hidden_layers {
        let q = role_map.get(&(TensorRole::AttentionQuery, Some(layer_idx)));
        let k = role_map.get(&(TensorRole::AttentionKey, Some(layer_idx)));
        match (q, k) {
            (Some(q), Some(k)) => {
                let q_out = projection_out_dim(q, hidden_size, "Q projection")?;
                let k_out = projection_out_dim(k, hidden_size, "K projection")?;
                layer_qk_dims.push((q_out, k_out));
            }
            _ => {}
        }
    }

    // Use the most frequent (q_out, k_out) as the default configuration
    let (q_out, k_out) = if layer_qk_dims.is_empty() {
        if num_hidden_layers > 0 {
            return Err(ModelConfigError::InvalidConfig(
                "cannot derive attention params: missing Q/K projection in layer 0".to_string(),
            ));
        }
        (0, 0)
    } else {
        // Count frequency of each (q_out, k_out) pair
        let mut freq: HashMap<(usize, usize), usize> = HashMap::new();
        for &dim in &layer_qk_dims {
            *freq.entry(dim).or_insert(0) += 1;
        }
        // Pick the most frequent
        *freq.iter().max_by_key(|(_, count)| *count).map(|(dim, _)| dim).unwrap()
    };

    // Cross-layer validation: all variants must share a common head_dim
    // (e.g., Gemma 4 sliding Q=2048 K=256 vs global Q=4096 K=512, both head_dim=256)
    if layer_qk_dims.len() > 1 {
        let unique_dims: Vec<(usize, usize)> = {
            let mut s = layer_qk_dims.clone();
            s.sort_unstable();
            s.dedup();
            s
        };
        if unique_dims.len() > 1 {
            // Verify all variants can share a head_dim with layer 0's head candidates
            for &(alt_q, alt_k) in &unique_dims {
                if (alt_q, alt_k) == (q_out, k_out) {
                    continue;
                }
                // At least one valid head_dim must divide both default and variant dims
                let shares_head_dim = VALID_HEAD_DIMS.iter().any(|hd| {
                    q_out % hd == 0 && k_out % hd == 0 && alt_q % hd == 0 && alt_k % hd == 0
                });
                if !shares_head_dim {
                    return Err(ModelConfigError::InvalidConfig(format!(
                        "cross-layer head_dim inconsistency: default ({}, {}) vs ({}, {}) share no valid head_dim",
                        q_out, k_out, alt_q, alt_k
                    )));
                }
            }
        }
    }

    let mut head_candidates = Vec::new();
    if q_out > 0 && k_out > 0 {
        for head_dim in VALID_HEAD_DIMS {
            if hints.head_dim.is_some_and(|hint| head_dim != hint) {
                continue;
            }
            if q_out % head_dim != 0 || k_out % head_dim != 0 {
                continue;
            }
            let n_head = q_out / head_dim;
            let n_kv = k_out / head_dim;

            // Basic sanity checks
            if n_head == 0 || n_kv == 0 {
                continue;
            }
            if n_head % n_kv != 0 {
                continue;
            } // Grouped Query Attention constraint

            head_candidates.push((n_head, n_kv, head_dim));
        }
        // If hint eliminated all candidates (e.g., GGUF rope_dimension_count
        // reports global-layer head_dim=512, but majority layers use head_dim=256),
        // retry without the hint.
        if head_candidates.is_empty() && hints.head_dim.is_some() {
            for head_dim in VALID_HEAD_DIMS {
                if q_out % head_dim != 0 || k_out % head_dim != 0 {
                    continue;
                }
                let n_head = q_out / head_dim;
                let n_kv = k_out / head_dim;
                if n_head == 0 || n_kv == 0 {
                    continue;
                }
                if n_head % n_kv != 0 {
                    continue;
                }
                head_candidates.push((n_head, n_kv, head_dim));
            }
        }
    } else if num_hidden_layers == 0 {
        // Embedding model without attention
        head_candidates.push((0, 0, 0));
    }

    head_candidates.sort_unstable_by_key(|&(_, _, hd)| hd);
    head_candidates.dedup();

    if head_candidates.is_empty() {
        return Err(ModelConfigError::InvalidConfig(format!(
            "cannot derive head_dim from tensors (q_out={q_out}, k_out={k_out})"
        )));
    }

    // Prefer the largest head_dim (most common in modern architectures).
    // GGUF/ONNX callers may override with metadata head_count afterwards.
    // SafeTensors callers get a unique candidate from config.json hints.
    let (num_attention_heads, num_key_value_heads, head_dim) = *head_candidates.last().unwrap();

    // 5. Intermediate Size
    let mut intermediate_size = None;
    if let Some(gate) = role_map.get(&(TensorRole::FfnGate, Some(0))) {
        let out = projection_out_dim(gate, hidden_size, "FFN Gate")?;
        intermediate_size = Some(out);
    } else if let Some(up) = role_map.get(&(TensorRole::FfnUp, Some(0))) {
        let out = projection_out_dim(up, hidden_size, "FFN Up")?;
        intermediate_size = Some(out);
    }

    let dtype = derive_dtype(&metas)?;

    // 6. Build Tensor Map
    // We only need to store the pattern for each role once.
    let mut tensor_map = HashMap::new();
    for ((role, layer), meta) in &role_map {
        if tensor_map.contains_key(role) {
            continue;
        }

        let pattern = if let Some(idx) = layer {
            anonymize_layer_index(&meta.name, *idx)
        } else {
            meta.name.clone()
        };
        tensor_map.insert(*role, pattern);
    }

    Ok(TensorDerivedConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        num_hidden_layers,
        intermediate_size,
        vocab_size,
        head_dim,
        dtype,
        tensor_map,
    })
}

