//! Layer 1: Manifest system (SSOT).

pub mod types;

pub use types::{
    map_architecture_token, tensor_rules_for_arch, FileMap, MoEConfig, ModelArchitecture,
    ModelKind, ModelManifest, RouterType, TensorNamingRule, TensorRole, EMPTY_FILE_MAP,
};
