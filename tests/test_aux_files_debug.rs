//! Test GGUF aux files

use gllm::loader::Loader;

#[test]
fn test_gguf_aux_files() {
    let loader = Loader::from("Mungert/SmolLM2-135M-Instruct-GGUF").expect("Failed to load");
    assert!(
        !loader.aux_files().is_empty(),
        "expected at least one auxiliary file"
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
