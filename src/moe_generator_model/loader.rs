//! MoE Generator Model Weight Loader - Pure L3 API compatible.

use crate::moe_generator_model::{MoEExpertWeights, MoEGeneratorModel, MoELayerWeights};
use crate::parallel_parser::{
    is_shard_index, load_embedding, load_linear, parse_shards, take_cached_shards, LoadConfig,
    ShardedTensorLoader, TensorLoader,
};
use crate::types::{Error, Result};
use crate::weight_loader::mappings::Architecture as MappingArchitecture;
use crate::weight_loader::shards::ShardIndex;
use crate::weight_loader::WeightLoader;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

impl MoEGeneratorModel {
    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        if is_shard_index(safetensors_path) {
            return self.load_sharded_safetensors(safetensors_path);
        }

        let file = File::open(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to map SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let loader = WeightLoader::from_bytes(&mmap)?;
        self.load_from_loader(&loader)
    }

    pub fn load_awq(&mut self, path: &Path) -> Result<()> {
        self.load_safetensors(path)
    }

    pub fn load_gguf(&mut self, _path: &Path) -> Result<()> {
        Err(Error::LoadError(
            "GGUF loading is not supported for MoE generator models".into(),
        ))
    }

    fn load_sharded_safetensors(&mut self, index_path: &Path) -> Result<()> {
        let model_dir = index_path.parent().ok_or_else(|| {
            Error::LoadError("Shard index path is missing parent directory".into())
        })?;
        let index = ShardIndex::from_index_file(index_path)?;
        let parsed = match take_cached_shards(model_dir) {
            Some(parsed) => parsed,
            None => parse_shards(index.shard_paths(model_dir), &LoadConfig::default())?,
        };
        let loader = ShardedTensorLoader::new(&parsed, &index)?;
        self.load_from_loader(&loader)
    }

    fn load_from_loader<L: TensorLoader + Sync>(&mut self, loader: &L) -> Result<()> {
        let architecture = match self.architecture {
            crate::registry::Architecture::GptOss => MappingArchitecture::GptOss,
            crate::registry::Architecture::GLM4 | crate::registry::Architecture::GLM4MoE => MappingArchitecture::Glm,
            _ => MappingArchitecture::Llama,
        };

        // Load embedding weights
        let embed_names = architecture.embedding_weight_candidates();
        let expected_embed_len = self.embedding_mut().len();
        let mut embed_loaded = false;
        for name in embed_names {
            if loader.has_tensor(name) {
                let embedding = load_embedding(loader, name)?;
                if embedding.data.len() != expected_embed_len {
                    return Err(Error::LoadError(format!(
                        "Embedding weight {} has {} elements, expected {}",
                        name,
                        embedding.data.len(),
                        expected_embed_len
                    )));
                }
                self.embedding_mut().copy_from_slice(&embedding.data);
                embed_loaded = true;
                break;
            }
        }
        if !embed_loaded {
            return Err(Error::LoadError(
                "Embedding weights not found for MoE generator model".into(),
            ));
        }

        // Load layer weights
        // Parallel loading using Rayon (ARCH-LOADER-PARALLEL)
        // This is critical for MoE models where fused weight splitting is CPU intensive
        self.layers.par_iter_mut().enumerate().try_for_each(|(layer_idx, layer)| {
            let mut last_err = None;
            let mut layer_loaded = false;
            for prefix in architecture.layer_prefix_candidates(layer_idx) {
                match load_layer_weights(loader, &prefix, layer_idx, layer) {
                    Ok(()) => {
                        layer_loaded = true;
                        break;
                    }
                    Err(err) => {
                        if is_missing_layer_weights_error(&err) {
                            last_err = Some(err);
                            continue;
                        }
                        return Err(err);
                    }
                }
            }
            if !layer_loaded {
                return Err(last_err.unwrap_or_else(|| {
                    Error::LoadError(format!("Layer weights not found for layer {}", layer_idx))
                }));
            }
            Ok(())
        })?;

        // Load final norm weights
        let final_norm_names = architecture.final_norm_weight_candidates();
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_data = norm_tensor.into_weight_vector()?.data;
                let final_norm = self.final_norm_mut();
                if norm_data.len() != final_norm.len() {
                    return Err(Error::LoadError(format!(
                        "Final norm weight {} has {} elements, expected {}",
                        name,
                        norm_data.len(),
                        final_norm.len()
                    )));
                }
                final_norm.copy_from_slice(&norm_data);
                break;
            }
        }

        // Load LM head weights
        let lm_head_names = architecture.lm_head_weight_candidates();
        let expected_lm_head_len = self.lm_head_mut().len();
        let mut lm_head_loaded = false;
        for name in lm_head_names {
            if loader.has_tensor(name) {
                let lm_head = load_embedding(loader, name)?;
                if lm_head.data.len() != expected_lm_head_len {
                    return Err(Error::LoadError(format!(
                        "LM head weight {} has {} elements, expected {}",
                        name,
                        lm_head.data.len(),
                        expected_lm_head_len
                    )));
                }
                self.lm_head_mut().copy_from_slice(&lm_head.data);
                lm_head_loaded = true;
                break;
            }
        }
        if !lm_head_loaded {
            // Tie embeddings to LM head if not found
            let embedding = self.embedding_mut().to_vec();
            self.lm_head_mut().copy_from_slice(&embedding);
        }

        Ok(())
    }
}

fn is_missing_layer_weights_error(err: &Error) -> bool {
    matches!(err, Error::LoadError(msg) if msg.contains("weights not found for prefix")
        || msg.contains("MoE gate weights not found")
        || msg.contains("MoE expert weights not found"))
}

fn load_layer_weights<L: TensorLoader + Sync>(
    loader: &L,
    prefix: &str,
    layer_idx: usize,
    layer: &mut MoELayerWeights,
) -> Result<()> {
    // Load attention projection weights (Q, K, V, O)
    load_attention_weights(loader, prefix, layer)?;

    // Load layer norms
    if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
        let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let norm_data = norm_tensor.into_weight_vector()?.data;
        if norm_data.len() != layer.input_norm.len() {
            return Err(Error::LoadError(format!(
                "Input norm weight has {} elements, expected {}",
                norm_data.len(),
                layer.input_norm.len()
            )));
        }
        layer.input_norm.copy_from_slice(&norm_data);
    }

    if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
        let norm_tensor = loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let norm_data = norm_tensor.into_weight_vector()?.data;
        if norm_data.len() != layer.post_attn_norm.len() {
            return Err(Error::LoadError(format!(
                "Post attention norm weight has {} elements, expected {}",
                norm_data.len(),
                layer.post_attn_norm.len()
            )));
        }
        layer.post_attn_norm.copy_from_slice(&norm_data);
    }

    // Load MoE weights (router + experts)
    load_moe_weights(loader, prefix, layer_idx, layer)?;

    Ok(())
}

fn load_attention_weights<L: TensorLoader>(
    loader: &L,
    prefix: &str,
    layer: &mut MoELayerWeights,
) -> Result<()> {
    // Try different attention weight naming conventions
    let attn_prefixes = [
        format!("{}.self_attn", prefix),
        format!("{}.attention", prefix),
    ];

    for attn_prefix in &attn_prefixes {
        let q_name = format!("{}.q_proj.weight", attn_prefix);
        if !loader.has_tensor(&q_name) {
            continue;
        }

        // Load Q projection
        let q_linear = load_linear(loader, &q_name, Some(&format!("{}.q_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&q_linear, &mut layer.q_weight)?;

        // Load K projection
        let k_name = format!("{}.k_proj.weight", attn_prefix);
        let k_linear = load_linear(loader, &k_name, Some(&format!("{}.k_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&k_linear, &mut layer.k_weight)?;

        // Load V projection
        let v_name = format!("{}.v_proj.weight", attn_prefix);
        let v_linear = load_linear(loader, &v_name, Some(&format!("{}.v_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&v_linear, &mut layer.v_weight)?;

        // Load O projection
        let o_name = format!("{}.o_proj.weight", attn_prefix);
        let o_linear = load_linear(loader, &o_name, Some(&format!("{}.o_proj.bias", attn_prefix)))?;
        copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;

        return Ok(());
    }

    // Try fused QKV (query_key_value or qkv_proj)
    for attn_prefix in &attn_prefixes {
        // pattern 1: query_key_value (GLM style)
        let qkv_names = [
            format!("{}.query_key_value.weight", attn_prefix),
            format!("{}.qkv_proj.weight", attn_prefix),
        ];

        for qkv_name in qkv_names {
            if loader.has_tensor(&qkv_name) {
                let qkv = load_linear(loader, &qkv_name, Some(&format!("{}.bias", qkv_name.strip_suffix(".weight").unwrap())))?;
                let qkv_data = qkv.as_dense_slice().ok_or_else(|| {
                    Error::LoadError("Fused QKV must be dense".into())
                })?;

                // Split Q, K, V
                // Assumption: standard splitting by hidden_size (or heads)
                // Q: [num_q_heads * head_dim, hidden]
                // K: [num_kv_heads * head_dim, hidden]
                // V: [num_kv_heads * head_dim, hidden]

                let _hidden_size = qkv.in_features();
                let _total_out = qkv.out_features();

                // Calculate split sizes based on layer config
                // We don't have head config passed here easily unless we look at layer buffers
                let q_size = layer.q_weight.len();
                let k_size = layer.k_weight.len();
                let v_size = layer.v_weight.len();

                if q_size + k_size + v_size != qkv_data.len() {
                    // Try to infer strict split if sizes don't match (e.g. if layer buffers pre-allocated differently)
                    // But here we must match the buffer sizes
                    return Err(Error::LoadError(format!(
                        "Fused QKV size mismatch: got {}, expected {} (q) + {} (k) + {} (v)",
                        qkv_data.len(), q_size, k_size, v_size
                    )));
                }

                // Split data
                layer.q_weight.copy_from_slice(&qkv_data[..q_size]);
                layer.k_weight.copy_from_slice(&qkv_data[q_size..q_size + k_size]);
                layer.v_weight.copy_from_slice(&qkv_data[q_size + k_size..]);

                // Load O projection
                let o_name = format!("{}.dense.weight", attn_prefix); // GLM style
                let o_name_alt = format!("{}.o_proj.weight", attn_prefix);

                let o_final = if loader.has_tensor(&o_name) {
                    o_name
                } else {
                    o_name_alt
                };

                let o_linear = load_linear(loader, &o_final, Some(&format!("{}.bias", o_final.strip_suffix(".weight").unwrap())))?;
                copy_linear_to_slice(&o_linear, &mut layer.o_weight)?;

                return Ok(());
            }
        }
    }

    Err(Error::LoadError(format!(
        "Attention weights not found for prefix {}",
        prefix
    )))
}

fn load_moe_weights<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer_idx: usize,
    layer: &mut MoELayerWeights,
) -> Result<()> {
    // Load router/gate weights
    let gate_candidates = [
        format!("{layer_prefix}.block_sparse_moe.gate.weight"),
        format!("{layer_prefix}.moe.gate.weight"),
        format!("{layer_prefix}.router.weight"),
        format!("{layer_prefix}.gate.weight"),
        format!("{layer_prefix}.mlp.gate.weight"), // Qwen3 MoE
        format!("{layer_prefix}.mlp.router.weight"), // GptOss
        format!("{layer_prefix}.feed_forward.gate.weight"),
        format!("{layer_prefix}.feed_forward.router.weight"),
    ];

    let expected_gate_len = layer.router_weight.len();
    let mut gate_loaded = false;
    for name in gate_candidates {
        if loader.has_tensor(&name) {
            let gate_weights = load_embedding(loader, &name)?;
            if gate_weights.data.len() != expected_gate_len {
                return Err(Error::LoadError(format!(
                    "MoE gate weight {} has {} elements, expected {}",
                    name,
                    gate_weights.data.len(),
                    expected_gate_len
                )));
            }
            layer.router_weight.copy_from_slice(&gate_weights.data);
            gate_loaded = true;
            break;
        }
    }
    if !gate_loaded {
        return Err(Error::LoadError(format!(
            "MoE gate weights not found for layer {layer_idx}"
        )));
    }

    // Load expert weights
    // Check for fused experts first (GPT-OSS style)
    if try_load_fused_all_experts(loader, layer_prefix, layer)? {
        return Ok(());
    }

    let num_experts = layer.experts.len();
    for expert_idx in 0..num_experts {
        load_expert_weights(loader, layer_prefix, layer_idx, expert_idx, &mut layer.experts[expert_idx])?;
    }

    Ok(())
}

fn try_load_fused_all_experts<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer: &mut MoELayerWeights,
) -> Result<bool> {
    let gate_up_name = format!("{layer_prefix}.mlp.experts.gate_up_proj.weight");
    let down_name = format!("{layer_prefix}.mlp.experts.down_proj.weight");

    // Check if standard weights exist (or their quantized variants)
    let has_gate_up = loader.has_tensor(&gate_up_name) ||
                      loader.has_tensor(&gate_up_name.replace(".weight", ""));

    // Check if split quantized weights exist (blocks/scales)
    let gate_up_blocks = format!("{layer_prefix}.mlp.experts.gate_up_proj_blocks");
    let has_gate_up_blocks = loader.has_tensor(&gate_up_blocks);

    if !has_gate_up && !has_gate_up_blocks {
        return Ok(false);
    }

    let (gate_up_data, down_data) = if has_gate_up_blocks {
        // Load split quantized (blocks + scales)
        let gate_up_base = format!("{layer_prefix}.mlp.experts.gate_up_proj");
        let down_base = format!("{layer_prefix}.mlp.experts.down_proj");

        let gate_up = load_split_quantized_data(loader, &gate_up_base)?;
        let down = load_split_quantized_data(loader, &down_base)?;
        (gate_up, down)
    } else {
        // Load fused tensors via standard load_linear (Dense or AWQ)
        let gate_up_linear = load_linear(loader, &gate_up_name, Some(&format!("{layer_prefix}.mlp.experts.gate_up_proj.bias")))?;

        // Dequantize/Get data
        let gate_up = match gate_up_linear {
            crate::weight_loader::LinearWeights::Dense { weight, .. } => weight.as_slice().to_vec(),
            crate::weight_loader::LinearWeights::Quantized { weight } => weight.dequantize(),
        };

        let down_linear = load_linear(loader, &down_name, Some(&format!("{layer_prefix}.mlp.experts.down_proj.bias")))?;
        let down = match down_linear {
            crate::weight_loader::LinearWeights::Dense { weight, .. } => weight.as_slice().to_vec(),
            crate::weight_loader::LinearWeights::Quantized { weight } => weight.dequantize(),
        };
        (gate_up, down)
    };

    // Dimensions
    let num_experts = layer.experts.len();
    if num_experts == 0 { return Ok(true); }

    // Check sizes
    // gate_up should be [num_experts * 2 * intermediate, hidden]
    // down should be [num_experts * hidden, intermediate] (stacked output)
    // OR [hidden, num_experts * intermediate] (stacked input)

    let expert_0 = &layer.experts[0];
    let intermediate = expert_0.gate.len() / expert_0.down.len() * expert_0.down.len() / expert_0.gate.len() * 0 + expert_0.gate.len() / (expert_0.down.len() / expert_0.gate.len());
    // Wait, simpler:
    // gate: [intermediate, hidden] -> len = inter * hidden
    // down: [hidden, intermediate] -> len = hidden * inter
    // So gate.len() == down.len()
    let one_expert_elems = expert_0.gate.len(); // intermediate * hidden

    // Total gate_up size should be num_experts * 2 * one_expert_elems
    let expected_gate_up = num_experts * 2 * one_expert_elems;
    if gate_up_data.len() != expected_gate_up {
        return Err(Error::LoadError(format!(
            "Fused gate_up size mismatch: got {}, expected {}",
            gate_up_data.len(),
            expected_gate_up
        )));
    }

    // Total down size should be num_experts * one_expert_elems
    let expected_down = num_experts * one_expert_elems;
    if down_data.len() != expected_down {
        return Err(Error::LoadError(format!(
            "Fused down size mismatch: got {}, expected {}",
            down_data.len(),
            expected_down
        )));
    }

    // Split and assign
    // gate_up layout: [expert_0_gate; expert_0_up; expert_1_gate; expert_1_up ...] ?
    // OR [expert_0_gate_up; expert_1_gate_up ...] where gate_up is interweaved?
    // Usually it's stacked along output dimension.
    // gate_up out_features = num_experts * 2 * intermediate.
    // So it's [expert_0_gate, expert_0_up, expert_1_gate, expert_1_up, ...]

    let gate_up_chunk_size = 2 * one_expert_elems;
    let down_chunk_size = one_expert_elems;

    for i in 0..num_experts {
        let expert = &mut layer.experts[i];

        // Gate/Up
        let gu_start = i * gate_up_chunk_size;
        let gu_end = gu_start + gate_up_chunk_size;
        let gu_slice = &gate_up_data[gu_start..gu_end];

        // Split gate and up
        let split_pt = one_expert_elems;
        expert.gate.copy_from_slice(&gu_slice[..split_pt]);
        expert.up.copy_from_slice(&gu_slice[split_pt..]);

        // Down
        // down layout: [expert_0_down; expert_1_down ...]
        // down out_features = hidden (shared?).
        // down in_features = intermediate.
        // If it's [experts * hidden, intermediate], then it's stacked output? No.
        // If it's [hidden, experts * intermediate], then it's stacked input.
        // Standard merge: W_down = [W_down_0, W_down_1, ...]
        // Matrix mult: x @ W_down.T
        // If W_down is [hidden, experts*intermediate], then W_down.T is [experts*intermediate, hidden].
        // This effectively sums results if we select correct input slice? No.
        //
        // In MoE, usually `down` projects back to hidden.
        // `results = sum(expert_out @ down_expert)`
        // If weights are fused, it's often `[hidden, num_experts * intermediate]`
        // which corresponds to `out_features=hidden`, `in_features=num_experts*intermediate`.
        // BUT, `load_linear` returns weights in `[out, in]` (row major).
        // So `down_data` would be `[hidden, num_experts * intermediate]`.
        // This means row 0 contains part of exp0, part of exp1...
        // This is column-interleaved.
        //
        // However, many frameworks stack `down` weights as `[num_experts, hidden, intermediate]` and flatten.
        // If flattened to `[num_experts * hidden, intermediate]`, then it's row-stacked.
        // Let's assume row-stacked (simple concatenation of expert weights) because that's how `gate_up` usually is.

        let d_start = i * down_chunk_size;
        let d_end = d_start + down_chunk_size;
        expert.down.copy_from_slice(&down_data[d_start..d_end]);
    }

    Ok(true)
}

fn load_expert_weights<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer_idx: usize,
    expert_idx: usize,
    expert: &mut MoEExpertWeights,
) -> Result<()> {
    let base_candidates = [
        format!("{layer_prefix}.block_sparse_moe.experts.{expert_idx}"),
        format!("{layer_prefix}.moe.experts.{expert_idx}"),
        format!("{layer_prefix}.mlp.experts.{expert_idx}"),
        format!("{layer_prefix}.experts.{expert_idx}"),
        format!("{layer_prefix}.feed_forward.experts.{expert_idx}"),
    ];

    for base in base_candidates {
        if try_load_w1_w2_w3(loader, &base, expert)? {
            return Ok(());
        }
        if try_load_proj_triple(loader, &base, expert)? {
            return Ok(());
        }
        if try_load_fused_gate_up(loader, &base, expert)? {
            return Ok(());
        }
    }

    Err(Error::LoadError(format!(
        "MoE expert weights not found for layer {layer_idx} expert {expert_idx}"
    )))
}

fn try_load_w1_w2_w3<L: TensorLoader>(
    loader: &L,
    base: &str,
    expert: &mut MoEExpertWeights,
) -> Result<bool> {
    let gate_name = format!("{base}.w1.weight");
    let up_name = format!("{base}.w3.weight");
    let down_name = format!("{base}.w2.weight");

    if !loader.has_tensor(&gate_name) {
        return Ok(false);
    }
    if !loader.has_tensor(&up_name) || !loader.has_tensor(&down_name) {
        return Ok(false);
    }

    let gate_linear = load_linear(loader, &gate_name, None)?;
    copy_linear_to_slice(&gate_linear, &mut expert.gate)?;

    let up_linear = load_linear(loader, &up_name, None)?;
    copy_linear_to_slice(&up_linear, &mut expert.up)?;

    let down_linear = load_linear(loader, &down_name, None)?;
    copy_linear_to_slice(&down_linear, &mut expert.down)?;

    Ok(true)
}

fn try_load_proj_triple<L: TensorLoader>(
    loader: &L,
    base: &str,
    expert: &mut MoEExpertWeights,
) -> Result<bool> {
    let gate_name = format!("{base}.gate_proj.weight");
    let up_name = format!("{base}.up_proj.weight");
    let down_name = format!("{base}.down_proj.weight");

    if !loader.has_tensor(&gate_name) {
        return Ok(false);
    }
    if !loader.has_tensor(&up_name) || !loader.has_tensor(&down_name) {
        return Ok(false);
    }

    let gate_linear = load_linear(loader, &gate_name, Some(&format!("{base}.gate_proj.bias")))?;
    copy_linear_to_slice(&gate_linear, &mut expert.gate)?;

    let up_linear = load_linear(loader, &up_name, Some(&format!("{base}.up_proj.bias")))?;
    copy_linear_to_slice(&up_linear, &mut expert.up)?;

    let down_linear = load_linear(loader, &down_name, Some(&format!("{base}.down_proj.bias")))?;
    copy_linear_to_slice(&down_linear, &mut expert.down)?;

    Ok(true)
}

fn try_load_fused_gate_up<L: TensorLoader>(
    loader: &L,
    base: &str,
    expert: &mut MoEExpertWeights,
) -> Result<bool> {
    let gate_up_name = format!("{base}.gate_up_proj.weight");

    if !loader.has_tensor(&gate_up_name) {
        return Ok(false);
    }

    let gate_up = load_linear(loader, &gate_up_name, Some(&format!("{base}.gate_up_proj.bias")))?;
    let gate_up_slice = gate_up.as_dense_slice().ok_or_else(|| {
        Error::LoadError("Fused gate_up must be dense".into())
    })?;

    // Load down projection
    let down_candidates = [
        format!("{base}.down_proj.weight"),
        format!("{base}.w2.weight"),
    ];

    let mut down_linear = None;
    for name in down_candidates {
        if loader.has_tensor(&name) {
            down_linear = Some(load_linear(loader, &name, Some(&format!("{base}.down_proj.bias")))?);
            break;
        }
    }

    let down_linear = down_linear.ok_or_else(|| Error::LoadError(format!("Down proj not found for {}", base)))?;
    copy_linear_to_slice(&down_linear, &mut expert.down)?;

    // Split gate_up into gate and up
    // The fused tensor is [gate; up] along output dimension
    // shape: [2 * intermediate_size, hidden_size]

    let total_len = gate_up_slice.len();
    if total_len % 2 != 0 {
        return Err(Error::LoadError("Fused gate_up length not even".into()));
    }
    let split_len = total_len / 2;

    if expert.gate.len() != split_len || expert.up.len() != split_len {
        return Err(Error::LoadError(format!(
            "Fused gate_up split size mismatch: got {}, expected gate/up size {}",
            split_len, expert.gate.len()
        )));
    }

    expert.gate.copy_from_slice(&gate_up_slice[..split_len]);
    expert.up.copy_from_slice(&gate_up_slice[split_len..]);

    Ok(true)
}

fn try_load_fused_experts<L: TensorLoader>(
    loader: &L,
    layer_prefix: &str,
    layer_idx: usize,
    expert_idx: usize,
    expert: &mut MoEExpertWeights,
) -> Result<bool> {
    // Only load the fused tensor once and cache it?
    // Current architecture iterates experts, so we would load the huge tensor N times.
    // Optimization: we could cache it, but loader doesn't have state passed down easily except 'loader'.
    // However, the OS page cache / mmap should make repeated reads somewhat efficient if we discard data.
    // BUT, 'load_linear' allocates a new Vec every time. This is bad for performance (loading 40GB N times).
    //
    // Since we can't easily change the architecture of `load_expert_weights` (called in loop),
    // we should try to detect if we can load specific slices from the loader?
    // `load_linear` loads the *entire* tensor.
    //
    // CRITICAL: We cannot load the full fused tensor for every expert.
    // We must change the loop structure in `load_moe_weights`.
    // But `load_expert_weights` is called by `load_moe_weights`.
    // We should modify `load_moe_weights` to handle fused loading there.

    // For now, return false here and handle it in `load_moe_weights`.
    Ok(false)
}

/// Copy LinearWeights data to a flat slice.
fn copy_linear_to_slice(linear: &crate::weight_loader::LinearWeights, dest: &mut [f32]) -> Result<()> {
    match linear {
        crate::weight_loader::LinearWeights::Dense { weight, .. } => {
            let src = weight.as_slice();
            if src.len() != dest.len() {
                return Err(Error::LoadError(format!(
                    "Linear weight size mismatch: got {}, expected {}",
                    src.len(),
                    dest.len()
                )));
            }
            dest.copy_from_slice(src);
        }
        crate::weight_loader::LinearWeights::Quantized { weight } => {
            let src = weight.dequantize();
            if src.len() != dest.len() {
                return Err(Error::LoadError(format!(
                    "Quantized linear weight size mismatch (dequantized): got {}, expected {}",
                    src.len(),
                    dest.len()
                )));
            }
            dest.copy_from_slice(&src);
        }
    }
    Ok(())
}

fn load_split_quantized_data<L: TensorLoader>(loader: &L, base_name: &str) -> Result<Vec<f32>> {
    let blocks_name = format!("{}_blocks", base_name);
    let scales_name = format!("{}_scales", base_name);

    if !loader.has_tensor(&blocks_name) || !loader.has_tensor(&scales_name) {
        return Err(Error::LoadError(format!(
            "Missing split quantized tensors for {}",
            base_name
        )));
    }

    // Load scales
    // Use load_raw_tensor to handle U8 scales (used in GPT-OSS) which load_tensor doesn't support
    let raw_scales = loader.load_raw_tensor(&scales_name)?;
    let scales = if raw_scales.dtype == safetensors::Dtype::U8 {
        raw_scales.data.iter().map(|&x| x as f32).collect::<Vec<f32>>()
    } else {
        crate::weight_loader::convert_to_f32_cow(&raw_scales.data, raw_scales.dtype)?.into_owned()
    };

    // Load blocks (raw bytes)
    let blocks_tensor = loader.load_raw_tensor(&blocks_name)?;
    // RawTensor is a struct with data: Vec<u8>, not an enum
    let blocks = blocks_tensor.data;

    if scales.is_empty() {
        return Ok(Vec::new());
    }

    let bytes_per_scale = blocks.len() / scales.len();
    let mut output = Vec::with_capacity(scales.len() * 32);

    match bytes_per_scale {
        16 => {
            // Q4_0: 1 scale per 32 elements (16 bytes)
            for (i, scale) in scales.iter().enumerate() {
                let start = i * 16;
                let chunk = &blocks[start..start + 16];
                for &byte in chunk {
                    let v0 = (byte & 0x0F) as i8 - 8;
                    let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
                    output.push(v0 as f32 * *scale);
                    output.push(v1 as f32 * *scale);
                }
            }
        }
        32 => {
            // Q8_0: 1 scale per 32 elements (32 bytes)
            for (i, scale) in scales.iter().enumerate() {
                let start = i * 32;
                let chunk = &blocks[start..start + 32];
                for &byte in chunk {
                    let val = byte as i8;
                    output.push(val as f32 * *scale);
                }
            }
        }
        _ => {
            return Err(Error::LoadError(format!(
                "Unsupported split quantization ratio: {} bytes per scale. Expected 16 (Q4) or 32 (Q8).",
                bytes_per_scale
            )));
        }
    }

    Ok(output)
}
