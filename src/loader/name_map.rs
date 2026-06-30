//! TensorNameMap: one-time external→canonical name mapping built at load time.

use std::collections::{HashMap, HashSet};

/// Bidirectional mapping between external tensor names (from safetensors/GGUF/ONNX)
/// and canonical internal names used by JIT engine and weight packing.
pub struct TensorNameMap {
    /// external name → canonical name
    mapping: HashMap<String, String>,
    /// canonical name → external name
    canonical_to_external: HashMap<String, String>,
}

/// Extract MTP depth index from an external tensor name.
///
/// Recognized patterns:
///   `model.mtp_head.{k}.weight` → Some(k)
///   `model.mtp.{k}.weight`      → Some(k)
///   `model.layers.{N}.mtp_proj.{k}.weight` → Some(k)
fn extract_mtp_depth(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    let segments: Vec<&str> = lower.split('.').collect();
    for (i, seg) in segments.iter().enumerate() {
        if *seg == "mtp_head" || *seg == "mtp" || *seg == "mtp_proj" {
            // The next segment should be the depth index
            if let Some(next) = segments.get(i + 1) {
                if let Ok(depth) = next.parse::<usize>() {
                    return Some(depth);
                }
            }
        }
    }
    None
}

impl TensorNameMap {
    /// Build from a slice of external tensor names.
    ///
    /// Uses `match_tensor_role()` from loader to classify each name,
    /// then `TensorRole::to_canonical_name()` to produce the canonical form.
    /// Also maps bias tensors: for `foo.weight` → canonical `X`, maps `foo.bias` → `X.bias`.
    ///
    /// When `model_kind` is `Some(ModelKind::Reranker)`, GGUF's `output.weight`
    /// (which is GGUF's rename of `score.weight`) is remapped from `lm_head` to
    /// `classifier`. Reranker models never have a true language model head — their
    /// "output" tensor is always a classification head with shape [num_labels, hidden]
    /// (typically [2, hidden]), not [vocab_size, hidden].
    pub fn build_from_names(names: &[String], model_kind: Option<crate::manifest::ModelKind>) -> Self {
        let mut mapping = HashMap::new();
        let mut canonical_to_external = HashMap::new();
        let name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

        // Pass 1: map non-bias weight tensors
        for name in names {
            if let Some((role, layer)) = super::match_tensor_role(name) {
                let canonical = if role == crate::manifest::types::TensorRole::MoESharedExpert {
                    // MoESharedExpert needs projection suffix: L{N}.shared_expert.{gate_proj,up_proj,down_proj}
                    // BCE-036: 支持 gate_proj/up_proj/down_proj 和 w1/w2/w3 两套命名约定
                    let proj = if name.contains("gate_proj")
                        || name.contains("gate.weight")
                        || name.ends_with(".w1.weight")
                        || name.ends_with(".w1.bias")
                    {
                        "gate_proj"
                    } else if name.contains("up_proj")
                        || name.contains("up.weight")
                        || name.ends_with(".w3.weight")
                        || name.ends_with(".w3.bias")
                    {
                        "up_proj"
                    } else if name.contains("down_proj")
                        || name.contains("down.weight")
                        || name.ends_with(".w2.weight")
                        || name.ends_with(".w2.bias")
                    {
                        "down_proj"
                    } else {
                        // GGUF shared expert uses mlp.gate_proj / mlp.up_proj / mlp.down_proj
                        // which match FfnGate/FfnUp/FfnDown, not MoESharedExpert.
                        // If we reach here, just use the base name.
                        ""
                    };
                    let base = role.to_canonical_name(layer);
                    if proj.is_empty() { base } else { format!("{}.{}", base, proj) }
                } else if role == crate::manifest::types::TensorRole::MtpProjection {
                    // MTP projection weights have a depth index after the MTP keyword.
                    // Extract the depth index to produce unique canonical names:
                    //   model.mtp_head.{k}.weight → mtp_proj.{k}
                    //   model.mtp.{k}.weight      → mtp_proj.{k}
                    //   model.layers.{N}.mtp_proj.{k}.weight → L{N}.mtp_proj.{k}
                    let depth = extract_mtp_depth(name);
                    let base = role.to_canonical_name(layer);
                    match depth {
                        Some(d) => format!("{}.{}", base, d),
                        None => base,
                    }
                } else {
                    role.to_canonical_name(layer)
                };
                canonical_to_external.entry(canonical.clone())
                    .or_insert_with(|| name.clone());
                mapping.insert(name.clone(), canonical);
            }
        }

        // Pass 1.5: MoE expert weights (not handled by match_tensor_role)
        // Patterns:
        //   SafeTensors: model.layers.{L}.experts.{E}.{gate_proj,up_proj,down_proj}.weight
        //   GGUF:        blk.{L}.ffn_gate_ex{E}.weight / ffn_up_ex{E}.weight / ffn_down_ex{E}.weight
        for name in names {
            if mapping.contains_key(name) {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            let segments: Vec<&str> = lower.split('.').collect();
            let (layer, expert, proj, _ext_end) = if let Some(pos) = segments.iter().position(|s| *s == "experts") {
                // SafeTensors: ...experts.{E}.{proj}.weight
                if pos + 3 < segments.len() {
                    if let (Ok(e), Some(proj_name)) = (segments[pos + 1].parse::<usize>(), segments.get(pos + 2)) {
                        // Find layer index before "experts"
                        let mut l_idx = None;
                        for i in (0..pos).rev() {
                            if let Ok(idx) = segments[i].parse::<usize>() {
                                if i > 0 && matches!(segments[i - 1], "layers" | "blk" | "blocks" | "h" | "layer") {
                                    l_idx = Some(idx);
                                    break;
                                }
                            }
                        }
                        if let Some(l) = l_idx {
                            let proj = match *proj_name {
                                "gate_proj" | "w1" | "gate" => "gate_proj",
                                "up_proj" | "w3" => "up_proj",
                                "down_proj" | "w2" => "down_proj",
                                "gate_up_proj" => "gate_proj",
                                _ => continue,
                            };
                            (l, e, proj, pos + 3)
                        } else { continue }
                    } else { continue }
                } else { continue }
            } else {
                // GGUF: blk.{L}.ffn_{gate,up,down}_ex{E}.weight
                let mut found = None;
                for i in 1..segments.len() {
                    let seg = segments[i];
                    if let Some(rest) = seg.strip_prefix("ffn_gate_ex") {
                        if let Ok(e) = rest.parse::<usize>() {
                            if i > 1 && segments[i - 1].parse::<usize>().is_ok() {
                                let l = segments[i - 1].parse::<usize>().unwrap();
                                found = Some((l, e, "gate_proj", i + 1));
                            }
                        }
                    } else if let Some(rest) = seg.strip_prefix("ffn_up_ex") {
                        if let Ok(e) = rest.parse::<usize>() {
                            if i > 1 {
                                let l = segments[i - 1].parse::<usize>().unwrap();
                                found = Some((l, e, "up_proj", i + 1));
                            }
                        }
                    } else if let Some(rest) = seg.strip_prefix("ffn_down_ex") {
                        if let Ok(e) = rest.parse::<usize>() {
                            if i > 1 {
                                let l = segments[i - 1].parse::<usize>().unwrap();
                                found = Some((l, e, "down_proj", i + 1));
                            }
                        }
                    }
                }
                match found {
                    Some(v) => v,
                    None => continue,
                }
            };
            let canonical = format!("L{}.expert.{}.{}", layer, expert, proj);
            canonical_to_external.entry(canonical.clone())
                .or_insert_with(|| name.clone());
            mapping.insert(name.clone(), canonical);
        }

        // Pass 2: map bias tensors (derived from weight canonical names)
        let bias_entries: Vec<(String, String, String)> = mapping.iter()
            .filter(|(ext_name, _)| ext_name.ends_with(".weight"))
            .filter_map(|(ext_name, canonical)| {
                let bias_ext = format!("{}bias", &ext_name[..ext_name.len() - 6]);
                if name_set.contains(bias_ext.as_str()) {
                    let bias_canonical = format!("{}.bias", canonical);
                    Some((bias_ext, bias_canonical, canonical.clone()))
                } else {
                    None
                }
            })
            .collect();

        for (bias_ext, bias_canonical, _canonical) in bias_entries {
            canonical_to_external.entry(bias_canonical.clone())
                .or_insert_with(|| bias_ext.clone());
            mapping.insert(bias_ext, bias_canonical);
        }

        // Tied embeddings: when lm_head canonical doesn't exist (no separate lm_head
        // tensor), map lm_head → embed's external name.
        // This applies regardless of tie_word_embeddings config — if there's no separate
        // lm_head tensor in the weight files, the model physically uses tied embeddings.
        // (GGUF models may not accurately report tie_word_embeddings.)
        if !canonical_to_external.contains_key("lm_head") {
            if let Some(embed_ext) = canonical_to_external.get("embed").cloned() {
                canonical_to_external.insert("lm_head".to_string(), embed_ext.clone());
                mapping.entry(embed_ext.clone()).or_insert_with(|| "lm_head".to_string());
            }
        }

        // ARCH-RERANKER-CLASSIFY: For Reranker models, GGUF renames `score.weight`
        // to `output.weight`, which maps to `lm_head`. But rerankers never have a
        // true language model head — their "output" tensor is always a classification
        // head with shape [num_labels, hidden] (typically [2, hidden]), not
        // [vocab_size, hidden]. Remap `lm_head` → `classifier` so the graph builder
        // and executor take the classification path instead of the generative path.
        //
        // IMPORTANT: Only remap when lm_head is a *separate* tensor (not tied to
        // embeddings). If lm_head is tied to embed (same external name), the model
        // lacks a classification head entirely — the GGUF converter omitted score.weight.
        // In this case, the generative path (lm_head tied to embed → vocab-sized logits)
        // is the only viable option, and remapping would break it by producing a
        // vocab-sized "classifier" output.
        if matches!(model_kind, Some(crate::manifest::ModelKind::Reranker)) {
            let embed_ext = canonical_to_external.get("embed").cloned();
            let lm_head_ext = canonical_to_external.get("lm_head").cloned();
            let is_separate_lm_head = lm_head_ext.as_ref() != embed_ext.as_ref();
            if is_separate_lm_head {
                // Remap canonical "lm_head" → "classifier" in the mapping dict.
                // Find all external names that currently map to "lm_head".
                let reranker_remap: Vec<String> = mapping.iter()
                    .filter(|(_, cn)| *cn == "lm_head")
                    .map(|(ext, _)| ext.clone())
                    .collect();
                for ext in reranker_remap {
                    mapping.insert(ext, "classifier".to_string());
                }
                // Update reverse mapping too
                if let Some(ext) = canonical_to_external.remove("lm_head") {
                    canonical_to_external.insert("classifier".to_string(), ext);
                }
            }
        }

        Self { mapping, canonical_to_external }
    }

    /// Look up canonical name for an external name.
    pub fn to_canonical(&self, external_name: &str) -> Option<&str> {
        self.mapping.get(external_name).map(|s| s.as_str())
    }

    /// Return all canonical names that an external name maps to.
    /// For tied embeddings, "token_embd.weight" maps to both "embed" and "lm_head".
    pub fn all_canonical_for(&self, external_name: &str) -> Vec<&str> {
        // Check direct mapping first
        let mut result = Vec::new();
        if let Some(cn) = self.mapping.get(external_name) {
            result.push(cn.as_str());
        }
        // Check if this external name is referenced by canonical_to_external
        // (for tied embedding aliases)
        for (canonical, ext) in &self.canonical_to_external {
            if ext == external_name
                && !result.contains(&canonical.as_str()) {
                    result.push(canonical.as_str());
                }
        }
        result
    }

    /// Look up external name for a canonical name.
    pub fn to_external(&self, canonical_name: &str) -> Option<&str> {
        self.canonical_to_external.get(canonical_name).map(|s| s.as_str())
    }

    /// Resolve canonical → external, falling back to the canonical name itself.
    /// Returns a String to avoid lifetime issues.
    pub fn resolve_external_to_string(&self, canonical_name: &str) -> String {
        self.canonical_to_external
            .get(canonical_name)
            .cloned()
            .unwrap_or_else(|| canonical_name.to_string())
    }

    /// Iterate over all (external, canonical) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.mapping.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_basic_mapping() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.k_proj.weight".into(),
            "model.layers.0.input_layernorm.weight".into(),
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.layers.0.mlp.up_proj.weight".into(),
            "model.layers.0.mlp.down_proj.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("model.layers.0.input_layernorm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("model.layers.0.mlp.gate_proj.weight"), Some("L0.gate_proj"));
        assert_eq!(map.to_canonical("model.norm.weight"), Some("final_norm"));
        assert_eq!(map.to_canonical("lm_head.weight"), Some("lm_head"));

        // Reverse lookup
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
        assert_eq!(map.to_external("L0.q_proj"), Some("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn test_gguf_basic_mapping() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "blk.0.attn_q.weight".into(),
            "blk.0.attn_k.weight".into(),
            "blk.0.attn_norm.weight".into(),
            "blk.0.ffn_gate.weight".into(),
            "blk.0.ffn_up.weight".into(),
            "blk.0.ffn_down.weight".into(),
            "output_norm.weight".into(),
            "output.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("token_embd.weight"), Some("embed"));
        assert_eq!(map.to_canonical("blk.0.attn_q.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("blk.0.attn_norm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("blk.0.ffn_gate.weight"), Some("L0.gate_proj"));
        assert_eq!(map.to_canonical("output_norm.weight"), Some("final_norm"));
        assert_eq!(map.to_canonical("output.weight"), Some("lm_head"));
    }

    #[test]
    fn test_moe_expert_safetensors() {
        let names: Vec<String> = vec![
            "model.layers.3.experts.0.gate_proj.weight".into(),
            "model.layers.3.experts.0.up_proj.weight".into(),
            "model.layers.3.experts.0.down_proj.weight".into(),
            "model.layers.3.experts.7.gate_proj.weight".into(),
            "model.layers.3.experts.7.up_proj.weight".into(),
            "model.layers.3.experts.7.down_proj.weight".into(),
            "model.layers.3.mlp.gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.3.experts.0.gate_proj.weight"), Some("L3.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.3.experts.0.up_proj.weight"), Some("L3.expert.0.up_proj"));
        assert_eq!(map.to_canonical("model.layers.3.experts.0.down_proj.weight"), Some("L3.expert.0.down_proj"));
        assert_eq!(map.to_canonical("model.layers.3.experts.7.gate_proj.weight"), Some("L3.expert.7.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.3.mlp.gate.weight"), Some("L3.moe_gate"));

        // Reverse
        assert_eq!(map.to_external("L3.expert.0.gate_proj"), Some("model.layers.3.experts.0.gate_proj.weight"));
        assert_eq!(map.to_external("L3.expert.7.down_proj"), Some("model.layers.3.experts.7.down_proj.weight"));
    }

    #[test]
    fn test_moe_expert_gguf() {
        let names: Vec<String> = vec![
            "blk.3.ffn_gate_ex0.weight".into(),
            "blk.3.ffn_up_ex0.weight".into(),
            "blk.3.ffn_down_ex0.weight".into(),
            "blk.3.ffn_gate_ex7.weight".into(),
            "blk.3.ffn_gate_inp.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.3.ffn_gate_ex0.weight"), Some("L3.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_up_ex0.weight"), Some("L3.expert.0.up_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_down_ex0.weight"), Some("L3.expert.0.down_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_gate_ex7.weight"), Some("L3.expert.7.gate_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_gate_inp.weight"), Some("L3.moe_gate"));
    }

    // BCE-036: Mixtral HF 原生 block_sparse_moe 命名变体回归测试
    #[test]
    fn moe_mixtral_block_sparse_moe_naming() {
        // Mixtral HF 原生命名: block_sparse_moe.experts.{E}.{w1,w2,w3} + block_sparse_moe.gate
        let names: Vec<String> = vec![
            "model.layers.0.block_sparse_moe.gate.weight".into(),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight".into(),
            "model.layers.0.block_sparse_moe.experts.0.w2.weight".into(),
            "model.layers.0.block_sparse_moe.experts.0.w3.weight".into(),
            "model.layers.0.block_sparse_moe.experts.1.w1.weight".into(),
            "model.layers.0.block_sparse_moe.experts.1.w2.weight".into(),
            "model.layers.0.block_sparse_moe.experts.1.w3.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // MoE gate (block_sparse_moe.gate → moe_gate)
        assert_eq!(map.to_canonical("model.layers.0.block_sparse_moe.gate.weight"), Some("L0.moe_gate"));

        // Experts w1/w2/w3 → gate_proj/up_proj/down_proj (Pass 1.5 handles this)
        assert_eq!(map.to_canonical("model.layers.0.block_sparse_moe.experts.0.w1.weight"), Some("L0.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.0.block_sparse_moe.experts.0.w2.weight"), Some("L0.expert.0.down_proj"));
        assert_eq!(map.to_canonical("model.layers.0.block_sparse_moe.experts.0.w3.weight"), Some("L0.expert.0.up_proj"));
        assert_eq!(map.to_canonical("model.layers.0.block_sparse_moe.experts.1.w1.weight"), Some("L0.expert.1.gate_proj"));

        // Reverse mapping
        assert_eq!(map.to_external("L0.moe_gate"), Some("model.layers.0.block_sparse_moe.gate.weight"));
        assert_eq!(map.to_external("L0.expert.0.gate_proj"), Some("model.layers.0.block_sparse_moe.experts.0.w1.weight"));
        assert_eq!(map.to_external("L0.expert.1.down_proj"), Some("model.layers.0.block_sparse_moe.experts.1.w2.weight"));
    }

    // BCE-036: Qwen/DeepSeek GGUF shared_expert 单数形式 + w1/w2/w3
    #[test]
    fn moe_shared_expert_singular_form() {
        let names: Vec<String> = vec![
            "model.layers.0.mlp.shared_expert.w1.weight".into(),
            "model.layers.0.mlp.shared_expert.w2.weight".into(),
            "model.layers.0.mlp.shared_expert.w3.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_expert.w1.weight"), Some("L0.shared_expert.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_expert.w2.weight"), Some("L0.shared_expert.down_proj"));
        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_expert.w3.weight"), Some("L0.shared_expert.up_proj"));
    }

    #[test]
    fn test_bias_mapping() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.bias"), Some("L0.q_proj.bias"));
    }

    #[test]
    fn test_shared_expert_mapping() {
        let names: Vec<String> = vec![
            "model.layers.3.mlp.shared_experts.gate_proj.weight".into(),
            "model.layers.3.mlp.shared_experts.up_proj.weight".into(),
            "model.layers.3.mlp.shared_experts.down_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.gate_proj.weight"), Some("L3.shared_expert.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.up_proj.weight"), Some("L3.shared_expert.up_proj"));
        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.down_proj.weight"), Some("L3.shared_expert.down_proj"));

        // Reverse
        assert_eq!(map.to_external("L3.shared_expert.gate_proj"), Some("model.layers.3.mlp.shared_experts.gate_proj.weight"));
    }

    #[test]
    fn test_mtp_projection_global_deepseek_v3() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.mtp_head.0.weight".into(),
            "model.mtp_head.1.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // DeepSeek V3 global MTP weights: model.mtp_head.{k}.weight → mtp_proj.{k}
        assert_eq!(map.to_canonical("model.mtp_head.0.weight"), Some("mtp_proj.0"));
        assert_eq!(map.to_canonical("model.mtp_head.1.weight"), Some("mtp_proj.1"));

        // Reverse
        assert_eq!(map.to_external("mtp_proj.0"), Some("model.mtp_head.0.weight"));
        assert_eq!(map.to_external("mtp_proj.1"), Some("model.mtp_head.1.weight"));
    }

    #[test]
    fn test_mtp_projection_global_qwen3() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.mtp.0.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Qwen3 global MTP: model.mtp.{k}.weight → mtp_proj.{k}
        assert_eq!(map.to_canonical("model.mtp.0.weight"), Some("mtp_proj.0"));
        assert_eq!(map.to_external("mtp_proj.0"), Some("model.mtp.0.weight"));
    }

    #[test]
    fn test_mtp_projection_per_layer() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.5.mtp_proj.0.weight".into(),
            "model.layers.5.mtp_proj.1.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Per-layer variant: model.layers.{N}.mtp_proj.{k}.weight → L{N}.mtp_proj.{k}
        assert_eq!(map.to_canonical("model.layers.5.mtp_proj.0.weight"), Some("L5.mtp_proj.0"));
        assert_eq!(map.to_canonical("model.layers.5.mtp_proj.1.weight"), Some("L5.mtp_proj.1"));

        // Reverse
        assert_eq!(map.to_external("L5.mtp_proj.0"), Some("model.layers.5.mtp_proj.0.weight"));
        assert_eq!(map.to_external("L5.mtp_proj.1"), Some("model.layers.5.mtp_proj.1.weight"));
    }

    #[test]
    fn test_extract_mtp_depth() {
        assert_eq!(super::extract_mtp_depth("model.mtp_head.0.weight"), Some(0));
        assert_eq!(super::extract_mtp_depth("model.mtp_head.2.weight"), Some(2));
        assert_eq!(super::extract_mtp_depth("model.mtp.1.weight"), Some(1));
        assert_eq!(super::extract_mtp_depth("model.layers.5.mtp_proj.3.weight"), Some(3));
        assert_eq!(super::extract_mtp_depth("model.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(super::extract_mtp_depth("model.norm.weight"), None);
    }

    // ── Additional coverage tests ──

    #[test]
    fn to_canonical_returns_none_for_unknown() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("nonexistent.weight"), None);
        assert_eq!(map.to_canonical(""), None);
    }

    #[test]
    fn to_external_returns_none_for_unknown() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_external("nonexistent"), None);
        assert_eq!(map.to_external(""), None);
    }

    #[test]
    fn resolve_external_to_string_falls_back() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        // Known canonical → returns external name
        assert_eq!(map.resolve_external_to_string("embed"), "model.embed_tokens.weight");
        // Unknown canonical → returns the canonical name itself
        assert_eq!(map.resolve_external_to_string("nonexistent"), "nonexistent");
    }

    #[test]
    fn tied_embeddings_without_lm_head() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            // No lm_head.weight — should tie to embed
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_external("lm_head"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn no_tied_embeddings_when_lm_head_exists() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        // lm_head exists separately, no tie
        assert_eq!(map.to_external("lm_head"), Some("lm_head.weight"));
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn all_canonical_for_single_mapping() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        let canonicals = map.all_canonical_for("model.embed_tokens.weight");
        assert!(canonicals.contains(&"embed"));
        assert_eq!(canonicals.len(), 1);
    }

    #[test]
    fn all_canonical_for_tied_embedding() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        let canonicals = map.all_canonical_for("model.embed_tokens.weight");
        // embed + lm_head (tied)
        assert!(canonicals.contains(&"embed"));
        assert!(canonicals.contains(&"lm_head"));
        assert_eq!(canonicals.len(), 2);
    }

    #[test]
    fn all_canonical_for_unknown_returns_empty() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        let canonicals = map.all_canonical_for("nonexistent");
        assert!(canonicals.is_empty());
    }

    #[test]
    fn iter_returns_all_pairs() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        let pairs: Vec<_> = map.iter().collect();
        assert!(pairs.len() >= 2);
        assert!(pairs.iter().any(|(ext, canon)| *ext == "model.embed_tokens.weight" && *canon == "embed"));
        assert!(pairs.iter().any(|(ext, canon)| *ext == "model.layers.0.self_attn.q_proj.weight" && *canon == "L0.q_proj"));
    }

    #[test]
    fn empty_names_produces_empty_map() {
        let map = TensorNameMap::build_from_names(&[], None);
        assert_eq!(map.to_canonical("anything"), None);
        assert_eq!(map.to_external("anything"), None);
        let pairs: Vec<_> = map.iter().collect();
        assert!(pairs.is_empty());
    }

    #[test]
    fn extract_mtp_depth_edge_cases() {
        // No depth after mtp_head
        assert_eq!(super::extract_mtp_depth("model.mtp_head.weight"), None);
        // Non-numeric depth
        assert_eq!(super::extract_mtp_depth("model.mtp_head.abc.weight"), None);
        // Mixed case (to_ascii_lowercase)
        assert_eq!(super::extract_mtp_depth("Model.MTP_Head.2.Weight"), Some(2));
        // Empty string
        assert_eq!(super::extract_mtp_depth(""), None);
    }

    #[test]
    fn gguf_moe_expert_w1_w2_w3_variants() {
        let names: Vec<String> = vec![
            "blk.0.ffn_gate_ex0.weight".into(),
            "blk.0.ffn_up_ex0.weight".into(),
            "blk.0.ffn_down_ex0.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("blk.0.ffn_gate_ex0.weight"), Some("L0.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("blk.0.ffn_up_ex0.weight"), Some("L0.expert.0.up_proj"));
        assert_eq!(map.to_canonical("blk.0.ffn_down_ex0.weight"), Some("L0.expert.0.down_proj"));
    }

    #[test]
    fn bias_without_weight_ignored() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.bias".into(),
            // No matching .weight → bias should not be mapped
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.bias"), None);
    }

    #[test]
    fn multiple_layers_mapped_correctly() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.1.self_attn.q_proj.weight".into(),
            "model.layers.15.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("model.layers.1.self_attn.q_proj.weight"), Some("L1.q_proj"));
        assert_eq!(map.to_canonical("model.layers.15.self_attn.q_proj.weight"), Some("L15.q_proj"));
    }

    #[test]
    fn gguf_output_norm_maps_to_final_norm() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "output_norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("output_norm.weight"), Some("final_norm"));
        assert_eq!(map.to_external("final_norm"), Some("output_norm.weight"));
    }

    // ── Additional tests: broader architecture coverage ──

    #[test]
    fn bert_encoder_attention_mapping() {
        // BERT/XLM-R encoder-style attention names
        let names: Vec<String> = vec![
            "encoder.layer.0.attention.self.query.weight".into(),
            "encoder.layer.0.attention.self.key.weight".into(),
            "encoder.layer.0.attention.self.value.weight".into(),
            "encoder.layer.0.attention.output.dense.weight".into(),
            "encoder.layer.0.attention.output.layernorm.weight".into(),
            "encoder.layer.0.output.layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("encoder.layer.0.attention.self.query.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("encoder.layer.0.attention.self.key.weight"), Some("L0.k_proj"));
        assert_eq!(map.to_canonical("encoder.layer.0.attention.self.value.weight"), Some("L0.v_proj"));
        assert_eq!(map.to_canonical("encoder.layer.0.attention.output.dense.weight"), Some("L0.o_proj"));
        assert_eq!(map.to_canonical("encoder.layer.0.attention.output.layernorm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("encoder.layer.0.output.layernorm.weight"), Some("L0.post_attn_norm"));
    }

    #[test]
    fn gpt2_wq_wk_wv_wo_mapping() {
        // GPT-2 style short names: wq, wk, wv, wo
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.wq.weight".into(),
            "model.layers.0.self_attn.wk.weight".into(),
            "model.layers.0.self_attn.wv.weight".into(),
            "model.layers.0.self_attn.wo.weight".into(),
            "model.layers.0.ffn_gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.wq.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.wk.weight"), Some("L0.k_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.wv.weight"), Some("L0.v_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.wo.weight"), Some("L0.o_proj"));
    }

    #[test]
    fn gemma_qk_norm_mapping() {
        // Gemma 4 QkNorm tensors: q_norm and k_norm
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_norm.weight".into(),
            "model.layers.0.self_attn.k_norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_norm.weight"), Some("L0.q_norm"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.k_norm.weight"), Some("L0.k_norm"));
        assert_eq!(map.to_external("L0.q_norm"), Some("model.layers.0.self_attn.q_norm.weight"));
        assert_eq!(map.to_external("L0.k_norm"), Some("model.layers.0.self_attn.k_norm.weight"));
    }

    #[test]
    fn classifier_and_score_mapping() {
        // XLM-R / BERT reranker: classifier.dense, classifier.out_proj, score
        let names: Vec<String> = vec![
            "encoder.layer.0.attention.self.query.weight".into(),
            "classifier.dense.weight".into(),
            "classifier.out_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("classifier.dense.weight"), Some("classifier.dense"));
        assert_eq!(map.to_canonical("classifier.out_proj.weight"), Some("classifier"));
    }

    #[test]
    fn score_tensor_maps_to_classifier() {
        // Some models use "score" as the classification output head
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "score.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("score.weight"), Some("classifier"));
        assert_eq!(map.to_external("classifier"), Some("score.weight"));
    }

    #[test]
    fn gguf_w1_w2_w3_ffn_mapping() {
        // GGUF w1/w2/w3 variants for FFN (not MoE experts)
        let names: Vec<String> = vec![
            "blk.0.ffn_gate.weight".into(),
            "blk.0.ffn_up.weight".into(),
            "blk.0.ffn_down.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.ffn_gate.weight"), Some("L0.gate_proj"));
        assert_eq!(map.to_canonical("blk.0.ffn_up.weight"), Some("L0.up_proj"));
        assert_eq!(map.to_canonical("blk.0.ffn_down.weight"), Some("L0.down_proj"));
    }

    #[test]
    fn gguf_ln1_ln2_norm_mapping() {
        // GGUF norm names: attn_norm, ffn_norm
        let names: Vec<String> = vec![
            "blk.0.attn_norm.weight".into(),
            "blk.0.ffn_norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.attn_norm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("blk.0.ffn_norm.weight"), Some("L0.post_attn_norm"));
    }

    #[test]
    fn bert_ln1_ln2_mapping() {
        // BERT layer_norm1 / layer_norm2
        let names: Vec<String> = vec![
            "encoder.layer.0.layer_norm1.weight".into(),
            "encoder.layer.0.layer_norm2.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("encoder.layer.0.layer_norm1.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("encoder.layer.0.layer_norm2.weight"), Some("L0.post_attn_norm"));
    }

    #[test]
    fn fused_qkv_proj_mapping() {
        // Fused QKV projection: self_attn.qkv_proj
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.qkv_proj.weight".into(),
            "model.layers.0.self_attn.o_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.qkv_proj.weight"), Some("L0.qkv_proj"));
        assert_eq!(map.to_external("L0.qkv_proj"), Some("model.layers.0.self_attn.qkv_proj.weight"));
    }

    #[test]
    fn position_embedding_and_rope_mapping() {
        // Position embedding (global) and rope freqs (global)
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.position_embedding.weight".into(),
            "model.rope.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.position_embedding.weight"), Some("position_embed"));
        assert_eq!(map.to_canonical("model.rope.weight"), Some("rope"));
    }

    #[test]
    fn moe_shared_expert_gguf_not_tied_to_experts() {
        // Shared expert gate weight is distinct from per-expert gate weights
        let names: Vec<String> = vec![
            "model.layers.2.mlp.shared_experts.gate_proj.weight".into(),
            "model.layers.2.experts.0.gate_proj.weight".into(),
            "model.layers.2.experts.1.gate_proj.weight".into(),
            "model.layers.2.mlp.gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Shared expert has its own canonical name
        assert_eq!(map.to_canonical("model.layers.2.mlp.shared_experts.gate_proj.weight"), Some("L2.shared_expert.gate_proj"));
        // Per-expert has different canonical names
        assert_eq!(map.to_canonical("model.layers.2.experts.0.gate_proj.weight"), Some("L2.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.2.experts.1.gate_proj.weight"), Some("L2.expert.1.gate_proj"));
        // Gate/router
        assert_eq!(map.to_canonical("model.layers.2.mlp.gate.weight"), Some("L2.moe_gate"));
    }

    #[test]
    fn moe_safetensors_expert_gate_variant() {
        // "gate" (short form) inside experts maps to gate_proj
        let names: Vec<String> = vec![
            "model.layers.1.experts.0.gate.weight".into(),
            "model.layers.1.experts.0.up_proj.weight".into(),
            "model.layers.1.experts.0.down_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.1.experts.0.gate.weight"), Some("L1.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.1.experts.0.up_proj.weight"), Some("L1.expert.0.up_proj"));
        assert_eq!(map.to_canonical("model.layers.1.experts.0.down_proj.weight"), Some("L1.expert.0.down_proj"));
    }

    #[test]
    fn moe_safetensors_gate_up_proj_fused_mapping() {
        // gate_up_proj fused variant inside experts
        let names: Vec<String> = vec![
            "model.layers.4.experts.2.gate_up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.4.experts.2.gate_up_proj.weight"), Some("L4.expert.2.gate_proj"));
    }

    #[test]
    fn mla_deepseek_v3_attention_mapping() {
        // DeepSeek V3 / R1 / Kimi-K2 MLA attention projections
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_a_proj.weight".into(),
            "model.layers.0.self_attn.q_b_proj.weight".into(),
            "model.layers.0.self_attn.kv_b_proj.weight".into(),
            "model.layers.0.self_attn.k_b_proj.weight".into(),
            "model.layers.0.self_attn.v_b_proj.weight".into(),
            "model.layers.0.self_attn.k_pe_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_a_proj.weight"), Some("L0.q_a_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_b_proj.weight"), Some("L0.q_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.kv_b_proj.weight"), Some("L0.kv_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.k_b_proj.weight"), Some("L0.k_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.v_b_proj.weight"), Some("L0.v_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.k_pe_proj.weight"), Some("L0.k_pe_proj"));
    }

    #[test]
    fn mtp_with_bias_tensor() {
        // MTP projection weight with a matching bias tensor
        let names: Vec<String> = vec![
            "model.mtp_head.0.weight".into(),
            "model.mtp_head.0.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.mtp_head.0.weight"), Some("mtp_proj.0"));
        // Bias should be derived: .weight→canonical, .bias→canonical.bias
        assert_eq!(map.to_canonical("model.mtp_head.0.bias"), Some("mtp_proj.0.bias"));
    }

    #[test]
    fn moe_expert_high_layer_and_expert_indices() {
        // High layer/expert indices ensure two-digit numbers work
        let names: Vec<String> = vec![
            "model.layers.59.experts.63.down_proj.weight".into(),
            "blk.59.ffn_down_ex63.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.59.experts.63.down_proj.weight"), Some("L59.expert.63.down_proj"));
        assert_eq!(map.to_canonical("blk.59.ffn_down_ex63.weight"), Some("L59.expert.63.down_proj"));
    }

    #[test]
    fn tied_embedding_with_gguf_output_weight() {
        // GGUF models: token_embd + output (no separate lm_head, should tie)
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "output_norm.weight".into(),
            // No output.weight → embed ties to lm_head
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("embed"), Some("token_embd.weight"));
        assert_eq!(map.to_external("lm_head"), Some("token_embd.weight"));
    }

    #[test]
    fn moe_expert_gguf_gate_up_down_all_projections() {
        // Full set of GGUF expert projections for a single expert
        let names: Vec<String> = vec![
            "blk.2.ffn_gate_ex3.weight".into(),
            "blk.2.ffn_up_ex3.weight".into(),
            "blk.2.ffn_down_ex3.weight".into(),
            "blk.2.ffn_gate_ex4.weight".into(),
            "blk.2.ffn_up_ex4.weight".into(),
            "blk.2.ffn_down_ex4.weight".into(),
            "blk.2.ffn_gate_inp.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Expert 3
        assert_eq!(map.to_canonical("blk.2.ffn_gate_ex3.weight"), Some("L2.expert.3.gate_proj"));
        assert_eq!(map.to_canonical("blk.2.ffn_up_ex3.weight"), Some("L2.expert.3.up_proj"));
        assert_eq!(map.to_canonical("blk.2.ffn_down_ex3.weight"), Some("L2.expert.3.down_proj"));
        // Expert 4
        assert_eq!(map.to_canonical("blk.2.ffn_gate_ex4.weight"), Some("L2.expert.4.gate_proj"));
        assert_eq!(map.to_canonical("blk.2.ffn_up_ex4.weight"), Some("L2.expert.4.up_proj"));
        assert_eq!(map.to_canonical("blk.2.ffn_down_ex4.weight"), Some("L2.expert.4.down_proj"));
        // Router
        assert_eq!(map.to_canonical("blk.2.ffn_gate_inp.weight"), Some("L2.moe_gate"));
    }

    #[test]
    fn canonical_to_external_first_wins_for_duplicate_roles() {
        // When multiple external names map to the same canonical, canonical_to_external
        // keeps the first one (or_insert_with).
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.word_embeddings.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Both should map to "embed" canonically
        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.to_canonical("model.word_embeddings.weight"), Some("embed"));
        // canonical_to_external keeps first-inserted
        let ext = map.to_external("embed");
        assert!(ext == Some("model.embed_tokens.weight") || ext == Some("model.word_embeddings.weight"));
    }

    // ── extract_mtp_depth exhaustive tests ──

    #[test]
    fn extract_mtp_depth_mtp_head_zero() {
        assert_eq!(super::extract_mtp_depth("model.mtp_head.0.weight"), Some(0));
    }

    #[test]
    fn extract_mtp_depth_mtp_head_large() {
        assert_eq!(super::extract_mtp_depth("model.mtp_head.99.weight"), Some(99));
    }

    #[test]
    fn extract_mtp_depth_mtp_short_form() {
        assert_eq!(super::extract_mtp_depth("model.mtp.5.weight"), Some(5));
    }

    #[test]
    fn extract_mtp_depth_per_layer_mtp_proj() {
        assert_eq!(super::extract_mtp_depth("model.layers.10.mtp_proj.3.weight"), Some(3));
    }

    #[test]
    fn extract_mtp_depth_no_mtp_keyword() {
        assert_eq!(super::extract_mtp_depth("model.layers.0.self_attn.q_proj.weight"), None);
    }

    #[test]
    fn extract_mtp_depth_keyword_at_end_no_next_segment() {
        assert_eq!(super::extract_mtp_depth("model.mtp_head"), None);
    }

    #[test]
    fn extract_mtp_depth_non_numeric_after_keyword() {
        assert_eq!(super::extract_mtp_depth("model.mtp_head.abc.weight"), None);
    }

    #[test]
    fn extract_mtp_depth_negative_not_recognized() {
        // Negative numbers are not valid usize, parse fails
        assert_eq!(super::extract_mtp_depth("model.mtp_head.-1.weight"), None);
    }

    #[test]
    fn extract_mtp_depth_mixed_case_uppercase() {
        assert_eq!(super::extract_mtp_depth("MODEL.MTP_HEAD.7.WEIGHT"), Some(7));
    }

    #[test]
    fn extract_mtp_depth_mixed_case_mixed() {
        assert_eq!(super::extract_mtp_depth("Model.Mtp_Head.3.Weight"), Some(3));
    }

    #[test]
    fn extract_mtp_depth_empty_string() {
        assert_eq!(super::extract_mtp_depth(""), None);
    }

    #[test]
    fn extract_mtp_depth_single_segment() {
        assert_eq!(super::extract_mtp_depth("mtp"), None);
    }

    #[test]
    fn extract_mtp_depth_mtp_proj_without_layer() {
        // "mtp_proj.4" without layers prefix — keyword still recognized
        assert_eq!(super::extract_mtp_depth("mtp_proj.4.weight"), Some(4));
    }

    #[test]
    fn extract_mtp_depth_first_keyword_wins() {
        // Multiple MTP keywords: first match wins
        let result = super::extract_mtp_depth("model.mtp.2.mtp_head.5.weight");
        assert_eq!(result, Some(2));
    }

    // ── TensorNameMap build_from_names boundary cases ──

    #[test]
    fn build_from_names_single_tensor() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn build_from_names_preserves_all_mappings_via_iter() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.k_proj.weight".into(),
            "model.layers.0.self_attn.v_proj.weight".into(),
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let pairs: std::collections::HashMap<&str, &str> =
            map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(pairs.len(), 7);
        assert_eq!(pairs.get("model.embed_tokens.weight"), Some(&"embed"));
        assert_eq!(pairs.get("model.layers.0.self_attn.q_proj.weight"), Some(&"L0.q_proj"));
        assert_eq!(pairs.get("lm_head.weight"), Some(&"lm_head"));
    }

    #[test]
    fn build_from_names_gguf_full_layer() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "blk.0.attn_q.weight".into(),
            "blk.0.attn_k.weight".into(),
            "blk.0.attn_v.weight".into(),
            "blk.0.attn_output.weight".into(),
            "blk.0.attn_norm.weight".into(),
            "blk.0.ffn_norm.weight".into(),
            "blk.0.ffn_gate.weight".into(),
            "blk.0.ffn_up.weight".into(),
            "blk.0.ffn_down.weight".into(),
            "output_norm.weight".into(),
            "output.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.attn_v.weight"), Some("L0.v_proj"));
        assert_eq!(map.to_canonical("blk.0.attn_output.weight"), Some("L0.o_proj"));
        assert_eq!(map.to_external("L0.o_proj"), Some("blk.0.attn_output.weight"));
    }

    // ── resolve_external_to_string coverage ──

    #[test]
    fn resolve_external_to_string_known_canonical() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.resolve_external_to_string("embed"), "model.embed_tokens.weight");
        assert_eq!(map.resolve_external_to_string("lm_head"), "lm_head.weight");
    }

    #[test]
    fn resolve_external_to_string_unknown_returns_input() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.resolve_external_to_string("does_not_exist"), "does_not_exist");
    }

    #[test]
    fn resolve_external_to_string_empty_input() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.resolve_external_to_string(""), "");
    }

    // ── to_canonical / to_external edge cases ──

    #[test]
    fn to_canonical_empty_map() {
        let map = TensorNameMap::build_from_names(&[], None);
        assert_eq!(map.to_canonical(""), None);
        assert_eq!(map.to_canonical("anything"), None);
    }

    #[test]
    fn to_external_empty_map() {
        let map = TensorNameMap::build_from_names(&[], None);
        assert_eq!(map.to_external(""), None);
        assert_eq!(map.to_external("embed"), None);
    }

    #[test]
    fn to_canonical_input_not_in_mapping() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        // A name that exists but was not recognized by match_tensor_role
        assert_eq!(map.to_canonical("model.unknown_tensor.weight"), None);
    }

    // ── all_canonical_for edge cases ──

    #[test]
    fn all_canonical_for_empty_map() {
        let map = TensorNameMap::build_from_names(&[], None);
        let result = map.all_canonical_for("anything");
        assert!(result.is_empty());
    }

    #[test]
    fn all_canonical_for_empty_name() {
        let names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        let map = TensorNameMap::build_from_names(&names, None);
        let result = map.all_canonical_for("");
        assert!(result.is_empty());
    }

    #[test]
    fn all_canonical_for_no_duplicates_in_single_mapping() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // embed maps only to "embed", not to "lm_head" since lm_head has its own tensor
        let canonicals = map.all_canonical_for("model.embed_tokens.weight");
        assert!(canonicals.contains(&"embed"));
        assert_eq!(canonicals.len(), 1);
    }

    // ── Bias tensor edge cases ──

    #[test]
    fn bias_mapping_multiple_weights_with_bias() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
            "model.layers.0.self_attn.k_proj.weight".into(),
            "model.layers.0.self_attn.k_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.bias"), Some("L0.q_proj.bias"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.k_proj.bias"), Some("L0.k_proj.bias"));
        assert_eq!(map.to_external("L0.q_proj.bias"), Some("model.layers.0.self_attn.q_proj.bias"));
        assert_eq!(map.to_external("L0.k_proj.bias"), Some("model.layers.0.self_attn.k_proj.bias"));
    }

    #[test]
    fn bias_without_matching_weight_not_mapped() {
        // Only bias present, no matching weight → bias not mapped
        let names: Vec<String> = vec![
            "model.layers.0.mlp.gate_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("model.layers.0.mlp.gate_proj.bias"), None);
    }

    // ── Tied embedding edge cases ──

    #[test]
    fn tied_embedding_lm_head_points_to_embed_external() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // lm_head canonical → embed's external name
        assert_eq!(map.to_external("lm_head"), Some("model.embed_tokens.weight"));
        // embed canonical → same external name
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn tied_embedding_all_canonical_includes_both() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("model.embed_tokens.weight");
        assert!(canonicals.contains(&"embed"));
        assert!(canonicals.contains(&"lm_head"));
        assert_eq!(canonicals.len(), 2);
    }

    #[test]
    fn no_tie_when_lm_head_has_own_tensor() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // embed and lm_head point to different external names
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
        assert_eq!(map.to_external("lm_head"), Some("lm_head.weight"));
    }

    // ── MoE expert GGUF with layer-zero and high-index ──

    #[test]
    fn moe_gguf_layer_zero_expert_zero() {
        let names: Vec<String> = vec![
            "blk.0.ffn_gate_ex0.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.ffn_gate_ex0.weight"), Some("L0.expert.0.gate_proj"));
    }

    // ── MoE expert safetensors w1/w2/w3 variant ──

    #[test]
    fn moe_safetensors_w1_w2_w3_mapped_by_pass_1() {
        // BCE-036: 在 MoE experts 上下文下，w1/w2/w3 必须映射到 expert-specific canonical，
        // 而不是被 dense FFN 的单 segment w1/w2/w3 模式抢先映射到 layer-level gate_proj。
        // 旧实现（BCE-036 之前）由于 SUFFIX_PATTERNS 中 `w1`/`w2`/`w3` 单 segment 模式
        // 抢先于 Pass 1.5 的 MoE expert 处理，错误地映射成 dense FFN。
        // BCE-036 通过 is_moe_context 守卫修复：experts/shared_expert/block_sparse_moe
        // 上下文下跳过单 segment w1/w2/w3，让 Pass 1.5 正确处理。
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.w1.weight".into(),
            "model.layers.0.experts.0.w3.weight".into(),
            "model.layers.0.experts.0.w2.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // w1 → gate_proj, w3 → up_proj, w2 → down_proj（Mixtral HF 约定）
        assert_eq!(map.to_canonical("model.layers.0.experts.0.w1.weight"), Some("L0.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.0.experts.0.w3.weight"), Some("L0.expert.0.up_proj"));
        assert_eq!(map.to_canonical("model.layers.0.experts.0.w2.weight"), Some("L0.expert.0.down_proj"));
    }

    // ── Shared expert up.weight / down.weight variant ──

    #[test]
    fn shared_expert_with_proj_suffix_recognized() {
        // The shared expert logic in build_from_names checks for "up_proj" / "down_proj" /
        // "gate_proj" or "gate.weight" / "up.weight" / "down.weight" suffixes.
        // With _proj suffix: standard path.
        let names: Vec<String> = vec![
            "model.layers.1.mlp.shared_experts.up_proj.weight".into(),
            "model.layers.1.mlp.shared_experts.down_proj.weight".into(),
            "model.layers.1.mlp.shared_experts.gate_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.1.mlp.shared_experts.up_proj.weight"), Some("L1.shared_expert.up_proj"));
        assert_eq!(map.to_canonical("model.layers.1.mlp.shared_experts.down_proj.weight"), Some("L1.shared_expert.down_proj"));
        assert_eq!(map.to_canonical("model.layers.1.mlp.shared_experts.gate_proj.weight"), Some("L1.shared_expert.gate_proj"));
    }

    #[test]
    fn shared_expert_gate_weight_not_recognized_without_proj_suffix() {
        // match_tensor_role only recognizes "shared_experts.gate_proj" not "shared_experts.gate".
        // Without the _proj suffix, the tensor is not matched at all.
        let names: Vec<String> = vec![
            "model.layers.1.mlp.shared_experts.gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // No suffix pattern matches "mlp.shared_experts.gate" → None
        assert_eq!(map.to_canonical("model.layers.1.mlp.shared_experts.gate.weight"), None);
    }

    // ── High layer indices ──

    #[test]
    fn high_layer_index_mapping() {
        let names: Vec<String> = vec![
            "model.layers.127.self_attn.q_proj.weight".into(),
            "blk.255.attn_q.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.127.self_attn.q_proj.weight"), Some("L127.q_proj"));
        assert_eq!(map.to_canonical("blk.255.attn_q.weight"), Some("L255.q_proj"));
    }

    // ── iter() returns empty for empty map ──

    #[test]
    fn iter_empty_map() {
        let map = TensorNameMap::build_from_names(&[], None);
        let count = map.iter().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn iter_count_matches_to_canonical_entries() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let iter_count = map.iter().count();
        // embed, L0.q_proj, L0.gate_proj, final_norm = 4 entries
        assert_eq!(iter_count, 4);
    }

    // ── GGUF MoE expert: segment before expert must be numeric layer ──

    #[test]
    fn gguf_moe_expert_requires_numeric_layer_before() {
        // "blk" followed by non-numeric segment → should not match GGUF expert pattern
        let names: Vec<String> = vec![
            "blk.abc.ffn_gate_ex0.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("blk.abc.ffn_gate_ex0.weight"), None);
    }

    // ── Safetensors MoE expert: unrecognized proj name skipped ──

    #[test]
    fn safetensors_moe_expert_unrecognized_proj_skipped() {
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.unknown_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);
        assert_eq!(map.to_canonical("model.layers.0.experts.0.unknown_proj.weight"), None);
    }

    // ── GGUF MoE: ffn_down_ex and ffn_up_ex variants ──

    #[test]
    fn gguf_moe_ffn_down_ex_and_up_ex() {
        let names: Vec<String> = vec![
            "blk.1.ffn_down_ex2.weight".into(),
            "blk.1.ffn_up_ex2.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.1.ffn_down_ex2.weight"), Some("L1.expert.2.down_proj"));
        assert_eq!(map.to_canonical("blk.1.ffn_up_ex2.weight"), Some("L1.expert.2.up_proj"));
    }

    // ── DeepSeek V3 full MLA + MoE + MTP integration ──

    #[test]
    fn deepseek_v3_full_layer_mapping() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.self_attn.q_a_proj.weight".into(),
            "model.layers.0.self_attn.q_b_proj.weight".into(),
            "model.layers.0.self_attn.kv_b_proj.weight".into(),
            "model.layers.0.self_attn.k_pe_proj.weight".into(),
            "model.layers.0.mlp.gate.weight".into(),
            "model.layers.0.experts.0.gate_proj.weight".into(),
            "model.layers.0.experts.0.up_proj.weight".into(),
            "model.layers.0.experts.0.down_proj.weight".into(),
            "model.layers.0.mlp.shared_experts.gate_proj.weight".into(),
            "model.layers.0.mlp.shared_experts.up_proj.weight".into(),
            "model.layers.0.mlp.shared_experts.down_proj.weight".into(),
            "model.mtp_head.0.weight".into(),
            "model.mtp_head.1.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // MLA
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_a_proj.weight"), Some("L0.q_a_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_b_proj.weight"), Some("L0.q_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.kv_b_proj.weight"), Some("L0.kv_b_proj"));
        assert_eq!(map.to_canonical("model.layers.0.self_attn.k_pe_proj.weight"), Some("L0.k_pe_proj"));

        // MoE gate
        assert_eq!(map.to_canonical("model.layers.0.mlp.gate.weight"), Some("L0.moe_gate"));

        // MoE experts
        assert_eq!(map.to_canonical("model.layers.0.experts.0.gate_proj.weight"), Some("L0.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.0.experts.0.down_proj.weight"), Some("L0.expert.0.down_proj"));

        // Shared experts
        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_experts.gate_proj.weight"), Some("L0.shared_expert.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_experts.down_proj.weight"), Some("L0.shared_expert.down_proj"));

        // MTP
        assert_eq!(map.to_canonical("model.mtp_head.0.weight"), Some("mtp_proj.0"));
        assert_eq!(map.to_canonical("model.mtp_head.1.weight"), Some("mtp_proj.1"));

        // Global
        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.to_canonical("lm_head.weight"), Some("lm_head"));
        assert_eq!(map.to_canonical("model.norm.weight"), Some("final_norm"));
    }

    // ── Per-layer MTP with bias ──

    #[test]
    fn per_layer_mtp_with_bias() {
        let names: Vec<String> = vec![
            "model.layers.3.mtp_proj.0.weight".into(),
            "model.layers.3.mtp_proj.0.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.3.mtp_proj.0.weight"), Some("L3.mtp_proj.0"));
        assert_eq!(map.to_canonical("model.layers.3.mtp_proj.0.bias"), Some("L3.mtp_proj.0.bias"));
    }

    // ── MTP without extractable depth falls back to base name ──

    #[test]
    fn mtp_projection_without_depth_index() {
        // If match_tensor_role recognizes MtpProjection but extract_mtp_depth returns None
        // (e.g. no numeric segment after mtp keyword), falls back to base canonical
        let names: Vec<String> = vec![
            "model.mtp.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // mtp is recognized as MtpProjection at global level
        assert!(map.to_canonical("model.mtp.weight").is_some());
    }

    // ── Multiple GGUF layers mapped independently ──

    #[test]
    fn gguf_multiple_independent_layers() {
        let names: Vec<String> = vec![
            "blk.0.attn_q.weight".into(),
            "blk.1.attn_q.weight".into(),
            "blk.2.attn_q.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.attn_q.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("blk.1.attn_q.weight"), Some("L1.q_proj"));
        assert_eq!(map.to_canonical("blk.2.attn_q.weight"), Some("L2.q_proj"));
    }

    // ── BERT encoder full layer ──

    #[test]
    fn bert_full_layer_all_projections() {
        let names: Vec<String> = vec![
            "encoder.layer.0.attention.self.query.weight".into(),
            "encoder.layer.0.attention.self.key.weight".into(),
            "encoder.layer.0.attention.self.value.weight".into(),
            "encoder.layer.0.attention.output.dense.weight".into(),
            "encoder.layer.0.attention.output.layernorm.weight".into(),
            "encoder.layer.0.output.dense.weight".into(),
            "encoder.layer.0.output.layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("encoder.layer.0.attention.self.query.weight"), Some("L0.q_proj"));
        // BERT output.dense maps based on match_tensor_role classification
        assert_eq!(map.to_canonical("encoder.layer.0.output.dense.weight"), Some("L0.down_proj"));
        assert_eq!(map.to_canonical("encoder.layer.0.attention.output.layernorm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("encoder.layer.0.output.layernorm.weight"), Some("L0.post_attn_norm"));
    }

    // ── Non-matching tensor names produce None ──

    #[test]
    fn non_matching_tensor_name_produces_none() {
        let names: Vec<String> = vec![
            "some.random.tensor.weight".into(),
            "another.weird.name.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("some.random.tensor.weight"), None);
        assert_eq!(map.to_canonical("another.weird.name.weight"), None);
    }

    // ── GGUF model with both tied embed and gguf-specific names ──

    #[test]
    fn gguf_tied_embed_with_norm() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "output_norm.weight".into(),
            // No output.weight → embed ties to lm_head
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("embed"), Some("token_embd.weight"));
        assert_eq!(map.to_external("final_norm"), Some("output_norm.weight"));
        assert_eq!(map.to_external("lm_head"), Some("token_embd.weight"));
    }

    // ── all_canonical_for with tied embed has both canonicals ──

    #[test]
    fn all_canonical_for_tied_returns_embed_and_lm_head() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("token_embd.weight");
        assert!(canonicals.contains(&"embed"));
        assert!(canonicals.contains(&"lm_head"));
    }

    // ────────────────────────────────────────────────────
    // ~50 NEW TESTS — public API coverage expansion
    // ────────────────────────────────────────────────────

    // ── Group 1: build_from_names structural properties ──

    #[test]
    fn build_from_names_tie_word_embeddings_true_no_separate_lm_head() {
        // tie_word_embeddings=true with no separate lm_head tensor:
        // tied embedding logic is data-driven, not flag-driven.
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("lm_head"), Some("model.embed_tokens.weight"));
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn build_from_names_tie_word_embeddings_true_with_separate_lm_head() {
        // Even with tie_word_embeddings=true, if lm_head exists separately it wins.
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("lm_head"), Some("lm_head.weight"));
        assert_eq!(map.to_external("embed"), Some("model.embed_tokens.weight"));
    }

    #[test]
    fn build_from_names_no_embed_prevents_tied_lm_head() {
        // Without an embed tensor, no tied lm_head can be created.
        let names: Vec<String> = vec![
            "model.norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("lm_head"), None);
        assert_eq!(map.to_external("embed"), None);
    }

    #[test]
    fn build_from_names_duplicate_external_names_deduplicated() {
        // Duplicate external names in input: HashMap overwrites, no panic.
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.iter().count(), 1);
    }

    #[test]
    fn build_from_names_many_layers_iter_count() {
        // Verify iter count scales with layer count.
        let mut names: Vec<String> = vec!["model.embed_tokens.weight".into()];
        for i in 0..5 {
            names.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
        }
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.iter().count(), 6); // 1 embed + 5 q_proj
    }

    // ── Group 2: to_canonical comprehensive coverage ──

    #[test]
    fn to_canonical_output_layer_variant() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.output_layer.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.output_layer.weight"), Some("lm_head"));
    }

    #[test]
    fn to_canonical_final_layernorm_variant() {
        let names: Vec<String> = vec![
            "model.final_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.final_layernorm.weight"), Some("final_norm"));
    }

    #[test]
    fn to_canonical_post_layernorm_variant() {
        let names: Vec<String> = vec![
            "model.post_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.post_layernorm.weight"), Some("final_norm"));
    }

    #[test]
    fn to_canonical_word_embeddings_variant() {
        let names: Vec<String> = vec![
            "model.word_embeddings.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.word_embeddings.weight"), Some("embed"));
    }

    #[test]
    fn to_canonical_post_attention_layernorm_variant() {
        let names: Vec<String> = vec![
            "model.layers.0.post_attention_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.post_attention_layernorm.weight"),
            Some("L0.post_attn_norm")
        );
    }

    #[test]
    fn to_canonical_pre_feedforward_layernorm_variant() {
        let names: Vec<String> = vec![
            "model.layers.0.pre_feedforward_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.pre_feedforward_layernorm.weight"),
            Some("L0.input_norm")
        );
    }

    #[test]
    fn to_canonical_post_feedforward_layernorm_variant() {
        let names: Vec<String> = vec![
            "model.layers.0.post_feedforward_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.post_feedforward_layernorm.weight"),
            Some("L0.post_attn_norm")
        );
    }

    #[test]
    fn to_canonical_ln1_ln2_variants() {
        let names: Vec<String> = vec![
            "model.layers.0.ln_1.weight".into(),
            "model.layers.0.ln_2.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.ln_1.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("model.layers.0.ln_2.weight"), Some("L0.post_attn_norm"));
    }

    // ── Group 3: to_external comprehensive coverage ──

    #[test]
    fn to_external_unrecognized_canonical_none() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("L0.q_proj"), None);
        assert_eq!(map.to_external("nonexistent"), None);
    }

    #[test]
    fn to_external_moe_gate_canonical() {
        let names: Vec<String> = vec![
            "model.layers.0.mlp.gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_external("L0.moe_gate"), Some("model.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn to_external_moe_router_variant() {
        let names: Vec<String> = vec![
            "model.layers.0.mlp.router.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.mlp.router.weight"), Some("L0.moe_gate"));
        assert_eq!(map.to_external("L0.moe_gate"), Some("model.layers.0.mlp.router.weight"));
    }

    #[test]
    fn to_external_depthwise_conv_mapping() {
        let names: Vec<String> = vec![
            "model.layers.0.conv_module.depthwise_conv.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.conv_module.depthwise_conv.weight"),
            Some("L0.depthwise_conv")
        );
        assert_eq!(
            map.to_external("L0.depthwise_conv"),
            Some("model.layers.0.conv_module.depthwise_conv.weight")
        );
    }

    #[test]
    fn to_external_attention_sinks_mapping() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.sinks.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.self_attn.sinks.weight"),
            Some("L0.attn_sinks")
        );
    }

    // ── Group 4: resolve_external_to_string comprehensive ──

    #[test]
    fn resolve_external_to_string_for_layer_tensor() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.resolve_external_to_string("L0.q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn resolve_external_to_string_for_tied_lm_head() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.resolve_external_to_string("lm_head"), "token_embd.weight");
        assert_eq!(map.resolve_external_to_string("embed"), "token_embd.weight");
    }

    #[test]
    fn resolve_external_to_string_for_moe_expert() {
        let names: Vec<String> = vec![
            "model.layers.0.experts.5.gate_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.resolve_external_to_string("L0.expert.5.gate_proj"),
            "model.layers.0.experts.5.gate_proj.weight"
        );
    }

    #[test]
    fn resolve_external_to_string_returns_owned_string() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let result: String = map.resolve_external_to_string("nonexistent_canonical");
        assert_eq!(result, "nonexistent_canonical");
    }

    // ── Group 5: all_canonical_for comprehensive ──

    #[test]
    fn all_canonical_for_lm_head_with_separate_tensor() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("lm_head.weight");
        assert!(canonicals.contains(&"lm_head"));
        assert_eq!(canonicals.len(), 1);
    }

    #[test]
    fn all_canonical_for_moe_expert_tensor() {
        let names: Vec<String> = vec![
            "model.layers.1.experts.3.up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("model.layers.1.experts.3.up_proj.weight");
        assert!(canonicals.contains(&"L1.expert.3.up_proj"));
        assert_eq!(canonicals.len(), 1);
    }

    #[test]
    fn all_canonical_for_shared_expert_tensor() {
        let names: Vec<String> = vec![
            "model.layers.0.mlp.shared_experts.up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("model.layers.0.mlp.shared_experts.up_proj.weight");
        assert!(canonicals.contains(&"L0.shared_expert.up_proj"));
        assert_eq!(canonicals.len(), 1);
    }

    #[test]
    fn all_canonical_for_bias_tensor() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("model.layers.0.self_attn.q_proj.bias");
        assert!(canonicals.contains(&"L0.q_proj.bias"));
    }

    // ── Group 6: iter() comprehensive ──

    #[test]
    fn iter_includes_bias_entries() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let pairs: std::collections::HashMap<&str, &str> =
            map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(pairs.get("model.layers.0.self_attn.q_proj.weight"), Some(&"L0.q_proj"));
        assert_eq!(pairs.get("model.layers.0.self_attn.q_proj.bias"), Some(&"L0.q_proj.bias"));
    }

    #[test]
    fn iter_includes_tied_lm_head_entry() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // mapping should have embed entry; tied lm_head adds via canonical_to_external
        // but does NOT create a new mapping entry for embed_ext→lm_head if embed already exists.
        let has_embed = map.iter().any(|(_, canon)| *canon == "embed");
        assert!(has_embed);
    }

    #[test]
    fn iter_includes_moe_expert_entries() {
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.gate_proj.weight".into(),
            "model.layers.0.experts.1.up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let pairs: Vec<_> = map.iter().collect();
        assert!(pairs.iter().any(|(ext, canon)| {
            *ext == "model.layers.0.experts.0.gate_proj.weight" && *canon == "L0.expert.0.gate_proj"
        }));
        assert!(pairs.iter().any(|(ext, canon)| {
            *ext == "model.layers.0.experts.1.up_proj.weight" && *canon == "L0.expert.1.up_proj"
        }));
    }

    // ── Group 7: MoE expert Pass 1.5 edge cases ──

    #[test]
    fn moe_expert_safetensors_layers_prefix_variants() {
        // "blocks" and "h" are valid layer prefixes
        let names: Vec<String> = vec![
            "model.blocks.3.experts.0.gate_proj.weight".into(),
            "model.h.5.experts.2.up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.blocks.3.experts.0.gate_proj.weight"),
            Some("L3.expert.0.gate_proj")
        );
        assert_eq!(
            map.to_canonical("model.h.5.experts.2.up_proj.weight"),
            Some("L5.expert.2.up_proj")
        );
    }

    #[test]
    fn moe_expert_gguf_missing_layer_index_skipped() {
        // If segment before ffn_gate_exN is not numeric, GGUF expert not matched
        let names: Vec<String> = vec![
            "blk.notanumber.ffn_gate_ex0.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.notanumber.ffn_gate_ex0.weight"), None);
    }

    #[test]
    fn moe_expert_safetensors_w3_maps_to_up_proj() {
        // w3 variant in experts → up_proj (via Pass 1.5)
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.w3.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // w3 is matched by Pass 1 as FfnUp (via match_tensor_role), not Pass 1.5
        // because w3 is in SUFFIX_PATTERNS
        let canonical = map.to_canonical("model.layers.0.experts.0.w3.weight");
        assert!(canonical.is_some());
    }

    #[test]
    fn moe_expert_gate_up_proj_fused_in_safetensors() {
        // gate_up_proj in experts → gate_proj (Pass 1.5 recognizes it)
        let names: Vec<String> = vec![
            "model.layers.2.experts.0.gate_up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.2.experts.0.gate_up_proj.weight"),
            Some("L2.expert.0.gate_proj")
        );
    }

    #[test]
    fn moe_expert_zero_expert_index() {
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.down_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.experts.0.down_proj.weight"),
            Some("L0.expert.0.down_proj")
        );
    }

    // ── Group 8: FFN variant names ──

    #[test]
    fn ffn_mlp_fc1_fc2_mapping() {
        // BERT-style FFN: mlp.fc1 (up) and mlp.fc2 (down)
        let names: Vec<String> = vec![
            "encoder.layer.0.intermediate.dense.weight".into(),
            "encoder.layer.0.output.dense.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("encoder.layer.0.intermediate.dense.weight"),
            Some("L0.up_proj")
        );
        assert_eq!(
            map.to_canonical("encoder.layer.0.output.dense.weight"),
            Some("L0.down_proj")
        );
    }

    #[test]
    fn ffn_gate_up_proj_fused_non_expert() {
        // Non-expert fused gate_up_proj (e.g. Qwen-type)
        let names: Vec<String> = vec![
            "model.layers.0.mlp.gate_up_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // gate_up_proj matches FfnGate in SUFFIX_PATTERNS
        let canonical = map.to_canonical("model.layers.0.mlp.gate_up_proj.weight");
        assert!(canonical.is_some());
    }

    #[test]
    fn ffn_self_attention_query_key_value_fused() {
        // self_attention.query_key_value fused QKV
        let names: Vec<String> = vec![
            "model.layers.0.self_attention.query_key_value.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.layers.0.self_attention.query_key_value.weight");
        assert!(canonical.is_some());
    }

    // ── Group 9: Attention out_proj variant ──

    #[test]
    fn attention_out_proj_mapping() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.out_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.self_attn.out_proj.weight"),
            Some("L0.o_proj")
        );
        assert_eq!(
            map.to_external("L0.o_proj"),
            Some("model.layers.0.self_attn.out_proj.weight")
        );
    }

    // ── Group 10: GGUF attn_q_norm / attn_k_norm ──

    #[test]
    fn gguf_attn_q_norm_and_attn_k_norm() {
        let names: Vec<String> = vec![
            "blk.0.attn_q_norm.weight".into(),
            "blk.0.attn_k_norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.0.attn_q_norm.weight"), Some("L0.q_norm"));
        assert_eq!(map.to_canonical("blk.0.attn_k_norm.weight"), Some("L0.k_norm"));
    }

    // ── Group 11: Global patch_embed mapping ──

    #[test]
    fn global_patch_embed_proj_mapping() {
        let names: Vec<String> = vec![
            "model.patch_embed.proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.patch_embed.proj.weight"), Some("patch_embed"));
    }

    #[test]
    fn global_vision_tower_patch_embed_mapping() {
        let names: Vec<String> = vec![
            "model.vision_tower.patch_embed.proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.vision_tower.patch_embed.proj.weight"),
            Some("patch_embed")
        );
    }

    // ── Group 12: Global embeddings.position_embedding variant ──

    #[test]
    fn global_embeddings_position_embedding_mapping() {
        let names: Vec<String> = vec![
            "model.embeddings.position_embedding.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.embeddings.position_embedding.weight"),
            Some("position_embed")
        );
    }

    // ── Group 13: MTP depth extraction additional edge cases ──

    #[test]
    fn extract_mtp_depth_trailing_dot_only() {
        // "mtp_head." — no segment after keyword
        assert_eq!(super::extract_mtp_depth("mtp_head."), None);
    }

    #[test]
    fn extract_mtp_depth_depth_zero_is_valid() {
        // Depth 0 is a valid index
        assert_eq!(super::extract_mtp_depth("model.mtp_proj.0.weight"), Some(0));
    }

    #[test]
    fn extract_mtp_depth_multiple_dots_between_keyword_and_depth() {
        // Normal: "model.mtp_head.3.weight" → depth 3
        assert_eq!(super::extract_mtp_depth("model.mtp_head.3.weight"), Some(3));
    }

    #[test]
    fn extract_mtp_depth_mtp_head_with_float_like_segment() {
        // "mtp_head.3.5.weight" → 3 is parsed first (stops at next segment)
        assert_eq!(super::extract_mtp_depth("model.mtp_head.3.5.weight"), Some(3));
    }

    // ── Group 14: Shared expert up.weight / down.weight variant ──



    #[test]
    fn shared_expert_gate_weight_suffix_not_in_pattern_table() {
        // match_tensor_role only recognizes "shared_experts.gate_proj", not "shared_experts.gate".
        // Without the _proj suffix, the tensor is not matched as MoESharedExpert.
        let names: Vec<String> = vec![
            "model.layers.0.mlp.shared_experts.gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // "shared_experts.gate" is NOT in SUFFIX_PATTERNS → None
        assert_eq!(map.to_canonical("model.layers.0.mlp.shared_experts.gate.weight"), None);
    }

    // ── Group 15: Cross-format consistency ──

    #[test]
    fn safetensors_and_gguf_same_layer_different_external_same_canonical() {
        // Same canonical name from two different format conventions
        let st_names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let gguf_names: Vec<String> = vec![
            "blk.0.attn_q.weight".into(),
        ].into_iter().collect();

        let st_map = TensorNameMap::build_from_names(&st_names, None);
        let gguf_map = TensorNameMap::build_from_names(&gguf_names, None);

        assert_eq!(st_map.to_canonical("model.layers.0.self_attn.q_proj.weight"), Some("L0.q_proj"));
        assert_eq!(gguf_map.to_canonical("blk.0.attn_q.weight"), Some("L0.q_proj"));
        // Both produce the same canonical name
        assert_eq!(
            st_map.to_canonical("model.layers.0.self_attn.q_proj.weight"),
            gguf_map.to_canonical("blk.0.attn_q.weight")
        );
    }

    // ── Group 16: Bias edge cases ──

    #[test]
    fn bias_for_moe_expert_weight() {
        // Expert weight has bias → derived canonical bias
        let names: Vec<String> = vec![
            "model.layers.0.experts.0.gate_proj.weight".into(),
            "model.layers.0.experts.0.gate_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.experts.0.gate_proj.bias"),
            Some("L0.expert.0.gate_proj.bias")
        );
    }

    #[test]
    fn bias_tensor_without_weight_suffix_not_auto_mapped() {
        // A bias tensor name that doesn't correspond to any .weight tensor
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // No matching .weight → bias not auto-mapped
        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.bias"), None);
    }

    #[test]
    fn bias_for_norm_tensor_not_auto_created() {
        // Norm weights don't typically have biases auto-created
        let names: Vec<String> = vec![
            "model.layers.0.input_layernorm.weight".into(),
            "model.layers.0.input_layernorm.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // input_layernorm.weight maps to L0.input_norm
        // Bias is derived: L0.input_norm.bias
        assert_eq!(map.to_canonical("model.layers.0.input_layernorm.weight"), Some("L0.input_norm"));
        assert_eq!(map.to_canonical("model.layers.0.input_layernorm.bias"), Some("L0.input_norm.bias"));
    }

    // ── Group 17: Rope global mapping ──

    #[test]
    fn rope_global_tensor_mapping() {
        let names: Vec<String> = vec![
            "model.rope.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.rope.weight"), Some("rope"));
        assert_eq!(map.to_external("rope"), Some("model.rope.weight"));
    }

    // ── Group 18: Mixed architecture in same map ──

    #[test]
    fn mixed_safetensors_and_gguf_names_in_same_map() {
        // Unusual but valid: both formats in same name list
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "blk.0.attn_q.weight".into(),
            "model.layers.1.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.embed_tokens.weight"), Some("embed"));
        assert_eq!(map.to_canonical("blk.0.attn_q.weight"), Some("L0.q_proj"));
        assert_eq!(map.to_canonical("model.layers.1.self_attn.q_proj.weight"), Some("L1.q_proj"));
    }

    // ── Group 19: Multiple biases on different layers ──

    #[test]
    fn bias_multiple_layers_independent() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
            "model.layers.1.self_attn.q_proj.weight".into(),
            "model.layers.1.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.self_attn.q_proj.bias"), Some("L0.q_proj.bias"));
        assert_eq!(map.to_canonical("model.layers.1.self_attn.q_proj.bias"), Some("L1.q_proj.bias"));
        assert_ne!(
            map.to_canonical("model.layers.0.self_attn.q_proj.bias"),
            map.to_canonical("model.layers.1.self_attn.q_proj.bias")
        );
    }

    // ── Group 20: GGUF MoE expert with ffn_up_ex non-zero expert ──

    #[test]
    fn gguf_moe_ffn_up_ex_non_zero_expert() {
        let names: Vec<String> = vec![
            "blk.4.ffn_up_ex15.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("blk.4.ffn_up_ex15.weight"), Some("L4.expert.15.up_proj"));
    }

    // ── Group 21: Self-consistency: to_canonical ↔ to_external round-trip ──

    #[test]
    fn roundtrip_safetensors_q_proj() {
        let names: Vec<String> = vec![
            "model.layers.7.self_attn.q_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.layers.7.self_attn.q_proj.weight").unwrap();
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "model.layers.7.self_attn.q_proj.weight");
    }

    #[test]
    fn roundtrip_gguf_ffn_gate() {
        let names: Vec<String> = vec![
            "blk.2.ffn_gate.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("blk.2.ffn_gate.weight").unwrap();
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "blk.2.ffn_gate.weight");
    }

    #[test]
    fn roundtrip_moe_expert() {
        let names: Vec<String> = vec![
            "model.layers.3.experts.2.down_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.layers.3.experts.2.down_proj.weight").unwrap();
        assert_eq!(canonical, "L3.expert.2.down_proj");
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "model.layers.3.experts.2.down_proj.weight");
    }

    #[test]
    fn roundtrip_bias_tensor() {
        let names: Vec<String> = vec![
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.layers.0.mlp.gate_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.layers.0.mlp.gate_proj.bias").unwrap();
        assert_eq!(canonical, "L0.gate_proj.bias");
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "model.layers.0.mlp.gate_proj.bias");
    }

    // ── Group 22: resolve_external_to_string round-trip ──

    #[test]
    fn resolve_external_roundtrip_matches_to_external() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.layers.0.input_layernorm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // For each canonical, resolve_external_to_string should match to_external
        for (canonical, _) in map.canonical_to_external.iter().take(5) {
            let resolved = map.resolve_external_to_string(canonical);
            let via_to_external = map.to_external(canonical).unwrap().to_string();
            assert_eq!(resolved, via_to_external);
        }
    }

    // ── Group 23: Additional coverage for shared expert weight suffix variants ──

    #[test]
    fn shared_expert_up_dot_weight_suffix_not_matched_by_role() {
        // "up.weight" suffix variant: match_tensor_role does not recognize
        // "shared_experts.up" without _proj suffix → None from Pass 1
        let names: Vec<String> = vec![
            "model.layers.0.mlp.shared_experts.up.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.mlp.shared_experts.up.weight"),
            None
        );
    }

    #[test]
    fn shared_expert_down_dot_weight_suffix_not_matched_by_role() {
        // "down.weight" suffix variant: same — not matched without _proj
        let names: Vec<String> = vec![
            "model.layers.0.mlp.shared_experts.down.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("model.layers.0.mlp.shared_experts.down.weight"),
            None
        );
    }

    // ── Group 24: resolve_external_to_string on empty map ──

    #[test]
    fn resolve_external_to_string_empty_map_returns_input() {
        let map = TensorNameMap::build_from_names(&[], None);

        assert_eq!(map.resolve_external_to_string("anything"), "anything");
        assert_eq!(map.resolve_external_to_string(""), "");
    }

    // ── Group 25: Safetensors MoE expert with too few segments after experts ──

    #[test]
    fn safetensors_moe_expert_truncated_after_experts_skipped() {
        // Only 2 segments after "experts" (need at least 3: {E}.{proj}.weight)
        let names: Vec<String> = vec![
            "model.layers.0.experts.5.gate_proj".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // Too few segments → not matched by Pass 1.5
        assert_eq!(map.to_canonical("model.layers.0.experts.5.gate_proj"), None);
    }

    // ── Group 26: GGUF expert ffn_gate_ex at first segment position ──

    #[test]
    fn gguf_moe_expert_ffn_gate_at_segment_zero_skipped() {
        // ffn_gate_ex at position 0: i > 1 is false → not matched
        let names: Vec<String> = vec![
            "ffn_gate_ex0.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("ffn_gate_ex0.weight"), None);
    }

    // ── Group 27: build_from_names with tie_word_embeddings and empty names ──

    #[test]
    fn build_from_names_tie_word_embeddings_empty_names() {
        let map = TensorNameMap::build_from_names(&[], None);

        assert_eq!(map.to_canonical("anything"), None);
        assert_eq!(map.to_external("embed"), None);
        assert_eq!(map.to_external("lm_head"), None);
    }

    // ── Group 28: GGUF tied embed all_canonical_for includes both ──

    #[test]
    fn all_canonical_for_gguf_tied_embed_returns_embed_and_lm_head() {
        let names: Vec<String> = vec![
            "token_embd.weight".into(),
            "output_norm.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonicals = map.all_canonical_for("token_embd.weight");
        assert!(canonicals.contains(&"embed"));
        assert!(canonicals.contains(&"lm_head"));
        assert_eq!(canonicals.len(), 2);
    }

    // ── Group 29: Round-trip for shared expert ──

    #[test]
    fn roundtrip_shared_expert() {
        let names: Vec<String> = vec![
            "model.layers.2.mlp.shared_experts.gate_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.layers.2.mlp.shared_experts.gate_proj.weight").unwrap();
        assert_eq!(canonical, "L2.shared_expert.gate_proj");
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "model.layers.2.mlp.shared_experts.gate_proj.weight");
    }

    // ── Group 30: Round-trip for MTP projection ──

    #[test]
    fn roundtrip_mtp_projection() {
        let names: Vec<String> = vec![
            "model.mtp_head.0.weight".into(),
            "model.embed_tokens.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        let canonical = map.to_canonical("model.mtp_head.0.weight").unwrap();
        assert_eq!(canonical, "mtp_proj.0");
        let external = map.to_external(canonical).unwrap();
        assert_eq!(external, "model.mtp_head.0.weight");
    }

    // ── Group 31: Safetensors MoE expert with no layer prefix before experts ──

    #[test]
    fn safetensors_moe_expert_no_layer_prefix_skipped() {
        // "experts" present but no valid "layers.N" prefix before it
        let names: Vec<String> = vec![
            "model.foo.0.experts.0.gate_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        // "foo" is not a valid layer prefix → l_idx remains None → skip
        assert_eq!(map.to_canonical("model.foo.0.experts.0.gate_proj.weight"), None);
    }

    // ── Group 32: mlp.gate.weight vs mlp.gate_proj.weight both map to moe_gate ──

    #[test]
    fn mlp_gate_weight_and_mlp_gate_proj_weight_both_moe_gate() {
        let names_gate: Vec<String> = vec![
            "model.layers.0.mlp.gate.weight".into(),
        ].into_iter().collect();
        let names_gate_proj: Vec<String> = vec![
            "model.layers.0.mlp.gate_proj.weight".into(),
        ].into_iter().collect();
        let map_gate = TensorNameMap::build_from_names(&names_gate, None);
        let map_gate_proj = TensorNameMap::build_from_names(&names_gate_proj, None);

        assert_eq!(map_gate.to_canonical("model.layers.0.mlp.gate.weight"), Some("L0.moe_gate"));
        // gate_proj matches FfnGate, not MoEGate — they are different roles
        let gp_canonical = map_gate_proj.to_canonical("model.layers.0.mlp.gate_proj.weight");
        assert!(gp_canonical.is_some());
        // They should map to different canonicals
        assert_ne!(
            map_gate.to_canonical("model.layers.0.mlp.gate.weight"),
            gp_canonical
        );
    }

    // ── Group 33: GGUF MoE expert only ffn_down_ex (no gate/up) ──

    #[test]
    fn gguf_moe_expert_only_down_proj_mapped() {
        let names: Vec<String> = vec![
            "blk.1.ffn_down_ex3.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(
            map.to_canonical("blk.1.ffn_down_ex3.weight"),
            Some("L1.expert.3.down_proj")
        );
        // No other experts mapped
        assert_eq!(map.to_canonical("blk.1.ffn_gate_ex3.weight"), None);
    }

    // ── Group 34: Non-numeric expert index in safetensors MoE skipped ──

    #[test]
    fn safetensors_moe_expert_non_numeric_expert_index_skipped() {
        // Expert index "abc" is not parseable as usize → Pass 1.5 skips
        let names: Vec<String> = vec![
            "model.layers.0.experts.abc.gate_proj.weight".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, None);

        assert_eq!(map.to_canonical("model.layers.0.experts.abc.gate_proj.weight"), None);
    }
}
