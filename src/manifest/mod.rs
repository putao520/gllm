//! Layer 1: Manifest system (SSOT).

pub mod types;

pub use types::{
    map_architecture_token, ArchFamily, FileMap, MoEConfig, ModelArchitecture,
    ModelKind, ModelManifest, RouterType, TensorRole, EMPTY_FILE_MAP,
};
