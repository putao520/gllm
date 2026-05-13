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

impl TensorNameMap {
    /// Build from a slice of external tensor names.
    ///
    /// Uses `match_tensor_role()` from loader to classify each name,
    /// then `TensorRole::to_canonical_name()` to produce the canonical form.
    /// Also maps bias tensors: for `foo.weight` → canonical `X`, maps `foo.bias` → `X.bias`.
    pub fn build_from_names(names: &[String], tie_word_embeddings: bool) -> Self {
        let mut mapping = HashMap::new();
        let mut canonical_to_external = HashMap::new();
        let name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

        // Pass 1: map non-bias weight tensors
        for name in names {
            if let Some((role, layer)) = super::match_tensor_role(name) {
                let canonical = if role == crate::manifest::types::TensorRole::MoESharedExpert {
                    // MoESharedExpert needs projection suffix: L{N}.shared_expert.{gate_proj,up_proj,down_proj}
                    let proj = if name.contains("gate_proj") || name.contains("gate.weight") {
                        "gate_proj"
                    } else if name.contains("up_proj") || name.contains("up.weight") {
                        "up_proj"
                    } else if name.contains("down_proj") || name.contains("down.weight") {
                        "down_proj"
                    } else {
                        // GGUF shared expert uses mlp.gate_proj / mlp.up_proj / mlp.down_proj
                        // which match FfnGate/FfnUp/FfnDown, not MoESharedExpert.
                        // If we reach here, just use the base name.
                        ""
                    };
                    let base = role.to_canonical_name(layer);
                    if proj.is_empty() { base } else { format!("{}.{}", base, proj) }
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
            let (layer, expert, proj, ext_end) = if let Some(pos) = segments.iter().position(|s| *s == "experts") {
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
                            if i > 1 && matches!(segments[i - 1].parse::<usize>(), Ok(_)) {
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
            if ext == external_name {
                if !result.contains(&canonical.as_str()) {
                    result.push(canonical.as_str());
                }
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
        let map = TensorNameMap::build_from_names(&names, false);

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
        let map = TensorNameMap::build_from_names(&names, false);

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
        let map = TensorNameMap::build_from_names(&names, false);

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
        let map = TensorNameMap::build_from_names(&names, false);

        assert_eq!(map.to_canonical("blk.3.ffn_gate_ex0.weight"), Some("L3.expert.0.gate_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_up_ex0.weight"), Some("L3.expert.0.up_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_down_ex0.weight"), Some("L3.expert.0.down_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_gate_ex7.weight"), Some("L3.expert.7.gate_proj"));
        assert_eq!(map.to_canonical("blk.3.ffn_gate_inp.weight"), Some("L3.moe_gate"));
    }

    #[test]
    fn test_bias_mapping() {
        let names: Vec<String> = vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.self_attn.q_proj.bias".into(),
        ].into_iter().collect();
        let map = TensorNameMap::build_from_names(&names, false);

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
        let map = TensorNameMap::build_from_names(&names, false);

        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.gate_proj.weight"), Some("L3.shared_expert.gate_proj"));
        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.up_proj.weight"), Some("L3.shared_expert.up_proj"));
        assert_eq!(map.to_canonical("model.layers.3.mlp.shared_experts.down_proj.weight"), Some("L3.shared_expert.down_proj"));

        // Reverse
        assert_eq!(map.to_external("L3.shared_expert.gate_proj"), Some("model.layers.3.mlp.shared_experts.gate_proj.weight"));
    }
}
