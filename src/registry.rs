//! Layer 1: Manifest registry (static lookup).

use crate::manifest::{all_manifests, manifest_by_id, KnownModel, ModelManifest};

pub fn lookup(alias: &str) -> Option<&'static ModelManifest> {
    let alias = alias.trim();
    if alias.is_empty() {
        return None;
    }

    all_manifests().iter().copied().find(|manifest| {
        manifest
            .aliases
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(alias))
            || manifest.hf_repo.eq_ignore_ascii_case(alias)
            || manifest
                .model_scope_repo
                .map_or(false, |repo| repo.eq_ignore_ascii_case(alias))
    })
}

pub fn lookup_by_id(model: KnownModel) -> &'static ModelManifest {
    manifest_by_id(model)
}

pub fn all() -> &'static [&'static ModelManifest] {
    all_manifests()
}
