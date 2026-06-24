//! Unit test: verify match_tensor_role + name_map for Phi4-mini tensor names

use gllm::loader::{match_tensor_role, name_map::TensorNameMap};
use gllm::manifest::TensorRole;

#[test]
fn phi4_fused_qkv_role_mapping() {
    // Phi4-mini tensor name for fused QKV
    let name = "model.layers.0.self_attn.qkv_proj.weight";
    let result = match_tensor_role(name);
    eprintln!("[TEST] match_tensor_role({:?}) = {:?}", name, result);
    assert!(result.is_some(), "qkv_proj should be recognized");
    let (role, layer) = result.unwrap();
    assert_eq!(role, TensorRole::AttentionFusedQkv, "Expected AttentionFusedQkv, got {:?}", role);
    assert_eq!(layer, Some(0));

    // Verify canonical name
    let canonical = role.to_canonical_name(layer);
    eprintln!("[TEST] canonical = {:?}", canonical);
    assert_eq!(canonical, "L0.qkv_proj");
}

#[test]
fn phi4_fused_gate_up_role_mapping() {
    let name = "model.layers.0.mlp.gate_up_proj.weight";
    let result = match_tensor_role(name);
    eprintln!("[TEST] match_tensor_role({:?}) = {:?}", name, result);
    assert!(result.is_some(), "gate_up_proj should be recognized");
    let (role, layer) = result.unwrap();
    // gate_up_proj matches ["mlp", "gate_up_proj"] → FfnGate
    assert_eq!(role, TensorRole::FfnGate, "Expected FfnGate for gate_up_proj, got {:?}", role);
    assert_eq!(layer, Some(0));
}

#[test]
fn phi4_name_map_full() {
    let names: Vec<String> = vec![
        "model.embed_tokens.weight".into(),
        "model.layers.0.self_attn.qkv_proj.weight".into(),
        "model.layers.0.self_attn.o_proj.weight".into(),
        "model.layers.0.input_layernorm.weight".into(),
        "model.layers.0.post_attention_layernorm.weight".into(),
        "model.layers.0.mlp.gate_up_proj.weight".into(),
        "model.layers.0.mlp.down_proj.weight".into(),
        "model.norm.weight".into(),
        // NO lm_head.weight — tied to embed_tokens
    ];

    let map = TensorNameMap::build_from_names(&names, Some(gllm::manifest::ModelKind::Chat));

    // Check fused QKV mapping
    let qkv_canonical = map.to_canonical("model.layers.0.self_attn.qkv_proj.weight");
    eprintln!("[TEST] qkv_proj canonical = {:?}", qkv_canonical);
    assert_eq!(qkv_canonical, Some("L0.qkv_proj"), "Fused QKV should map to L0.qkv_proj");

    // Check reverse: L0.qkv_proj → external
    let qkv_ext = map.to_external("L0.qkv_proj");
    eprintln!("[TEST] L0.qkv_proj external = {:?}", qkv_ext);
    assert_eq!(qkv_ext, Some("model.layers.0.self_attn.qkv_proj.weight"));

    // Check tied embeddings: no lm_head tensor → should map to embed
    let lm_head_ext = map.to_external("lm_head");
    eprintln!("[TEST] lm_head external = {:?}", lm_head_ext);
    assert_eq!(lm_head_ext, Some("model.embed_tokens.weight"), "Tied lm_head should map to embed");

    // Check gate_up_proj
    let gate_up = map.to_canonical("model.layers.0.mlp.gate_up_proj.weight");
    eprintln!("[TEST] gate_up_proj canonical = {:?}", gate_up);
    // gate_up_proj matches FfnGate → L0.gate_proj
    assert_eq!(gate_up, Some("L0.gate_proj"), "gate_up_proj should map to L0.gate_proj");
}
