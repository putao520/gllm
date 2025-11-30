//! Common utilities for integration tests.
//! Most tests now use real models directly, so this file is minimal.

use gllm::{ClientConfig, Device};
use std::path::PathBuf;

/// Get default models directory.
pub(crate) fn get_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gllm")
        .join("models")
}

/// Get default client config.
pub(crate) fn get_config(device: Device) -> ClientConfig {
    ClientConfig {
        device,
        models_dir: get_models_dir(),
    }
}
