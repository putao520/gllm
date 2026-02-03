//! Layer 1: Manifest system (SSOT).

mod models;
pub mod types;

pub use types::{
    FileMap, KnownModel, MoEConfig, ModelArchitecture, ModelManifest, RouterType, TensorNamingRule,
    EMPTY_FILE_MAP,
};

pub fn all_manifests() -> &'static [&'static ModelManifest] {
    models::ALL_MANIFESTS
}

pub fn manifest_by_id(model: KnownModel) -> &'static ModelManifest {
    models::manifest_by_id(model)
}

impl KnownModel {
    pub fn manifest(self) -> &'static ModelManifest {
        manifest_by_id(self)
    }
}
