//! Test GGUF aux files

use gllm::loader::{Loader, LoaderConfig};

#[test]
fn test_gguf_aux_files() {
    let config = LoaderConfig::from_env();
    let loader =
        Loader::from_source_with_config("Mungert/SmolLM2-135M-Instruct-GGUF".to_string(), config)
            .expect("Failed to load");
    // GGUF models typically have weight_paths and may have config_path
    assert!(
        !loader.weight_paths().is_empty(),
        "expected at least one weight file"
    );
    if let Some(config_path) = loader.config_path() {
        assert!(
            config_path.exists(),
            "config path should point to an existing file"
        );
    } else {
        panic!("config path should exist for GGUF models");
    }
}
