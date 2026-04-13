//! Layer 1: Manifest system (SSOT).

pub mod types;

pub use types::{
    map_architecture_token, map_architecture_token_for_kind, ArchFamily, FileMap, MoEConfig,
    ModelKind, ModelManifest, RouterType, TensorRole, EMPTY_FILE_MAP,
};
