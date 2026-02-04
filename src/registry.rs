//! Layer 1: Manifest overrides registry (string ID lookup).

use crate::manifest::{ManifestOverride, ModelKind, EMPTY_FILE_MAP};

const OVERRIDES: &[ManifestOverride] = &[
    ManifestOverride {
        model_id: "Qwen/Qwen3-Embedding",
        file_map: EMPTY_FILE_MAP,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        kind: Some(ModelKind::Embedding),
    },
    ManifestOverride {
        model_id: "Qwen/Qwen3-Reranker",
        file_map: EMPTY_FILE_MAP,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        kind: Some(ModelKind::Reranker),
    },
    ManifestOverride {
        model_id: "BAAI/bge-reranker-v3",
        file_map: EMPTY_FILE_MAP,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        kind: Some(ModelKind::Reranker),
    },
    ManifestOverride {
        model_id: "BAAI/bge-reranker-v2-m3",
        file_map: EMPTY_FILE_MAP,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        kind: Some(ModelKind::Reranker),
    },
    ManifestOverride {
        model_id: "openai-community/gpt2",
        file_map: EMPTY_FILE_MAP,
        rope_base_override: None,
        max_context_override: Some(1024),
        moe_config: None,
        kind: None,
    },
];

pub fn lookup(model_id: &str) -> Option<&'static ManifestOverride> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return None;
    }

    OVERRIDES
        .iter()
        .find(|override_entry| override_entry.model_id.eq_ignore_ascii_case(model_id))
}
