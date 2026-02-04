//! Layer 1: Manifest system (SSOT).

pub mod types;

pub use types::{
    FileMap, ManifestOverride, MoEConfig, ModelArchitecture, ModelKind, ModelManifest, RouterType,
    TensorNamingRule, EMPTY_FILE_MAP,
};
