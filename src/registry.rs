//! Layer 1: Manifest registry (static lookup).

use crate::manifest::{all_manifests, manifest_by_id, KnownModel, ModelManifest};

pub fn lookup(model_id: &str) -> Option<&'static ModelManifest> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return None;
    }

    all_manifests().iter().copied().find(|manifest| {
        manifest
            .model_id
            .eq_ignore_ascii_case(model_id)
    })
}

pub fn lookup_by_id(model: KnownModel) -> &'static ModelManifest {
    manifest_by_id(model)
}

pub fn all() -> &'static [&'static ModelManifest] {
    all_manifests()
}
